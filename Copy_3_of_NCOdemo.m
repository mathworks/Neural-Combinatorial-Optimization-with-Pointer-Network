%% Neural Combinatorial Optimization (NCO) example, using a custom attention model
% # Traveling Salesman Problem (TSP) Solver Using NCO, with Reinforcement Learning
%
% ## Workflow Overview
%
% This MATLAB code implements a reinforcement learning approach to solve the Traveling Salesman Problem using a neural attention-based architecture. The workflow follows these key steps:
%
% 1. **Initialization and Setup**
%    - Define hyperparameters (locations count, hidden dimensions, batch size)
%    - Set learning rates and entropy coefficients
%    - Configure early stopping parameters
%
% 2. **Network Architecture**
%    - Define actor network with LSTM encoder and attention mechanism
%    - Define critic network for value estimation
%    - Initialize both networks with proper input dimensions
%
% 3. **Training Loop**
%    - Generate random TSP instances
%    - Run attention-based actor for tour sampling
%    - Compute tour lengths and corresponding rewards
%    - Use critic to estimate baseline values
%    - Calculate advantage (reward - baseline)
%    - Compute actor and critic losses
%    - Apply gradient clipping and parameter updates
%    - Track performance and apply early stopping
%
% 4. **Inference and Evaluation**
%    - Perform greedy decoding for test instances
%    - Compare with 2-opt heuristic solution
%    - Calculate optimality gap
%    - Visualize tours for comparison
%
% 5. **Helper Functions**
%    - Tour length calculation
%    - Gradient clipping
%    - Actor and critic loss functions
%    - Attention mechanism implementation
%    - 2-opt local search implementation
%
% The code leverages principles from the "Neural Combinatorial
% Optimization" approach, implementing pointer networks with attention for
% sequential decision-making in the TSP context.

% Copyright 2025, The MathWorks, Inc.

%% Hyperparameters
nCities    = 20;        % number of cities (Increase for a bigger problem)
hiddenSize = 64;        % LSTM hidden dimension
batchSize  = 8;         % minibatch size
numEpochs  = 1000;      % training epochs (Increase for harder problems)
gradThreshold = 2.0;    % L2 norm threshold for gradient clipping

% Separate LRs: Implement separate Adam updates for Actor and Critic
% with different learning rates. Often, a slightly lower LR
% for the Actor can be beneficial if the Critic is learning too fast.
learnRateA = 5e-3; % Actor LR
learnRateC = 1e-2; % Potentially higher Critic LR

% Tune Entropy Coefficient (entropyCoefficient):
% Decrease: Try 1e-3 or 1e-4.
% This encourages the actor to exploit sequences it finds promising more quickly.
% Increase: Try 0.05 or 0.1.
% This encourages more exploration if you suspect it got stuck too early.
% Observe the AvgReward plot closely.
entropyCoefficient = 0.01; % Coefficient for entropy bonus in actor loss

% Temperature parameter
% Controls softmax sharpness (1.0=standard, <1.0=sharper, >1.0=smoother)
% <1.0 means more exploitation, >1.0 means more exploration
temperature = 1.2;

% Record for annealing
initialTemperature    = temperature;
initialEntropyCoeff   = entropyCoefficient;

% Early Stopping Parameters
patiencePercentage = 0.30; % Stop if no improvement for X% of total epochs
patience = ceil(numEpochs * patiencePercentage);

% Supervised Pre-training Parameters
numPreTrainEpochs = 100;        % Number of supervised pre-training epochs
preTrainBatchSize = 16;         % Can use larger batch size for pre-training
preTrainLR = 1e-3;              % Supervised learning rate
% Teacher forcing schedule
teacherForcingInitial = 1.0;           % initial TF ratio
% Set to 0 for pure imitation, 1 for pure policy learning
preTrainSamples = 5000;         % Generate this many examples for pre-training

% Optional: For reproducible results, fix the random seed generator
rng(1);

if canUseGPU
    gpuDevice(1); % select GPU #1 (or omit index to use default)
end

%% 1) Define Pointer (Actor) Network (with K, V, Q projection layers)
layersActor = [
    sequenceInputLayer(2,'Name','in', 'Normalization','none')
    % Add embedding
    fullyConnectedLayer(hiddenSize, 'Name', 'fcEmb') % , 'BiasInitializer','narrow-normal')
    reluLayer('Name','embRelu')
    lstmLayer(hiddenSize,'OutputMode','sequence','Name','encLSTM')
    % Note: We will use encLSTM's output to manually compute the Keys,
    % Values, and Queries projections. We add FC layers here so their
    % weights are part of the network learnables.
    % For Key projection
    fullyConnectedLayer(hiddenSize,'Name','fcKey') % , 'BiasInitializer','narrow-normal')
    % For Value projection - Not used in current decoder, but keep for potential use
    fullyConnectedLayer(hiddenSize,'Name','fcValue') % , 'BiasInitializer','narrow-normal')
    % For Query projection
    fullyConnectedLayer(hiddenSize,'Name','fcQuery') % , 'BiasInitializer','narrow-normal')

    % Dummy Decoder LSTM Layer to hold correct InputWeights
    lstmLayer(hiddenSize, 'Name', 'decLSTM_params')
    ];

actorNet = dlnetwork(layersActor);

% Initialize network with dummy data to infer sizes - Corrected dims for CBT [C,B,T]
dummyX = dlarray(rand(2, batchSize, nCities,  'single'), 'CBT');
actorNet = initialize(actorNet, dummyX);
fprintf('Actor network initialized.\n');

%% 2) Define Critic Network (Value Baseline)
layersCritic = [
    % Input averaged context - Should be [Hidden, 1, Batch] -> 'CBT'
    sequenceInputLayer(hiddenSize,'Name','cIn', 'Normalization','none')
    fullyConnectedLayer(hiddenSize*2,'Name','cFC1') % , 'BiasInitializer','narrow-normal')
    reluLayer('Name','cRelu1')
    fullyConnectedLayer(hiddenSize,'Name','cFC2') % , 'BiasInitializer','narrow-normal')
    reluLayer('Name','cRelu2')
    fullyConnectedLayer(1,'Name','cOut') % , 'BiasInitializer','narrow-normal')
    ];

criticNet = dlnetwork(layersCritic);

% Initialize critic - Input needs format [Hidden, Batch, 1] 'CBT'
dummyCriticIn = dlarray(rand(hiddenSize, batchSize, 1, 'single'), 'CBT'); % Use CBT [C=hidden, B=batch, T=1]
criticNet = initialize(criticNet, dummyCriticIn);
fprintf('Critic network initialized.\n');

% Move networks to GPU if available:
if canUseGPU
    actorNet  = dlupdate(@gpuArray, actorNet);
    criticNet = dlupdate(@gpuArray, criticNet);
    fprintf("Moved networks to GPU.\n");
end

%% Pre-Training Data Generation
fprintf('Generating pre-training data using 2-opt heuristic...\n');

preTrainCoords = rand(preTrainSamples, nCities, 2, 'single');
preTrainTours = zeros(preTrainSamples, nCities, 'uint32');

% Generate 2-opt solutions for pre-training
for i = 1:preTrainSamples
    % Randomly select starting city for diversity
    startCity = randi(nCities);
    [tour, ~] = solveTSP_2opt(squeeze(preTrainCoords(i,:,:)), startCity);
    preTrainTours(i,:) = uint32(tour);

    % Show progress
    if mod(i, 100) == 0
        fprintf('  Generated %d/%d pre-training examples\n', i, preTrainSamples);
    end
end

%% Pre-Training Loop
fprintf('Starting supervised pre-training for %d epochs...\n', numPreTrainEpochs);
avgGradPre = []; avgSqGradPre = []; % For Adam optimizer
historyPreTrainLoss = zeros(1, numPreTrainEpochs);

% use dlaccelerate() to speed up deep learning function evaluation for
% custom training loop.
supervisedLoss_acc = dlaccelerate(@supervisedLoss);
clearCache(supervisedLoss_acc);

% Supervised pretraining runs with a teacher-forcing ratio that starts high
% (e.g. 1.0) and decays over epochs, gradually shifting from imitation to
% sampling

for epoch = 1:numPreTrainEpochs
    % Decay teacher-forcing ratio over epochs
    tfRatio = max(0.1, teacherForcingInitial * (1 - epoch/numPreTrainEpochs));
    % Shuffle and select mini-batch
    idxBatch = randperm(preTrainSamples, preTrainBatchSize);
    batchCoords = preTrainCoords(idxBatch, :, :);
    batchTours = preTrainTours(idxBatch, :);

    % Permute to [C, B, T] format for dlarray
    X = permute(batchCoords, [3 1 2]);
    if canUseGPU
        dlX = dlarray(gpuArray(single(X)), 'CBT');          % <— GPU
    else
        dlX = dlarray(X, 'CBT');
    end

    % Supervised loss calculation with teacher forcing
    [gradPre, lossPre] = dlfeval(supervisedLoss_acc, actorNet, dlX, batchTours, tfRatio);

    % Gradient clipping
    gradPre = thresholdL2Norm(gradPre, gradThreshold);

    % Update network
    [actorNet, avgGradPre, avgSqGradPre] = adamupdate(actorNet, gradPre, avgGradPre, avgSqGradPre, epoch, preTrainLR);

    % Store and display progress
    historyPreTrainLoss(epoch) = double(gather(lossPre));
    if mod(epoch, 10) == 0 || epoch == 1
        fprintf('  Pre-train Epoch %d/%d, Loss=%.4f\n', epoch, numPreTrainEpochs, historyPreTrainLoss(epoch));
    end
end

% Plot pre-training loss
figure;
plot(1:numPreTrainEpochs, historyPreTrainLoss);
title('Supervised Pre-training Loss');
xlabel('Epoch');
ylabel('Cross-Entropy Loss');
grid on;

fprintf('Pre-training complete! Starting reinforcement learning...\n');

%% 3) Training Loop
% --- At the beginning of the training loop, add: ---
fprintf('Starting reinforcement learning after pre-training (nCities=%d, hiddenSize=%d, batchSize=%d)...\n', ...
    nCities, hiddenSize, batchSize);

% Reset Adam optimizer state for RL training
avgGradA = []; avgSqGradA = [];
avgGradC = []; avgSqGradC = [];

% --- Early Stopping Parameters ---
bestReward = -inf; % Initialize best reward seen so far (maximize reward, so start low)
epochsSinceImprovement = 0;
bestActorNetState = []; % Store learnables of the best model
bestCriticNetState = [];% Store learnables of the best model
% --------------------------------

fprintf('Starting training (nCities=%d, hiddenSize=%d, batchSize=%d, gradClip=%.1f, entropyCoeff=%.3f)...\n', ...
    nCities, hiddenSize, batchSize, gradThreshold, entropyCoefficient);
fprintf('Early stopping patience: %d epochs\n', patience);
tic; % Start timer

historyReward = zeros(1,numEpochs); % Keep track of rewards

% use dlaccelerate() to speed up deep learning function evaluation for
% custom training loop.
actorLoss_acc = dlaccelerate(@actorLoss);
clearCache(actorLoss_acc);
criticLoss_acc = dlaccelerate(@criticLoss);
clearCache(criticLoss_acc);

for epoch = 1:numEpochs
    % Anneal temperature & entropy coefficient ---
    temperature        = max(0.5, initialTemperature * (1 - epoch/numEpochs));
    entropyCoefficient = initialEntropyCoeff  * (1 - epoch/numEpochs);

    % Generate random TSP batch [B, N, C] -> [64, 20, 2]
    coords = rand(batchSize, nCities, 2, 'single');
    % Permute to [C, B, T] -> [2, 64, 20] for dlarray
    X = permute(coords,[3 1 2]);
    if canUseGPU
        dlX = dlarray(gpuArray(single(X)), 'CBT');          % <— GPU
    else
        dlX = dlarray(X, 'CBT');
    end

    % --- Actor Forward Pass & Sampling  ---
    [tours, logProbs, entropy] = actorStep(actorNet, dlX, temperature);
    % [tours, logProbs, entropy] = actorStep_coords(actorNet, dlX, temperature, coords);

    % --- Compute Rewards ---
    lengths = computeLengths(coords, tours);

    % --- Crossing penalty on reward ---
    crossings = zeros(batchSize,1);
    for b = 1:batchSize
        crossings(b) = countCrossings(double(tours(b,:)), squeeze(coords(b,:,:)));
    end

    lambda = 0.1;  % penalty weight (tune as needed)
    bonusFactor = 0.2; % bonus for better tours compared to batch average

    % Curriculum learning
    if epoch <= numEpochs * 0.3 %#ok<*UNRCH>
        rewards = -lengths - lambda * crossings;  % penalize crossings
    else
        % Add bonus for better tours compared to batch average
        avgLength = mean(lengths);

        rewards = -lengths - lambda * crossings + bonusFactor * max(0, (avgLength - lengths));
    end

    % --- OPTIONAL: Reward Normalization ---
    useNormalizedReward = true;

    if useNormalizedReward
        rewards_mean = mean(rewards);
        rewards_std = std(rewards, 1); % Use population std dev
        if rewards_std > 1e-8
            normalized_rewards = (rewards - rewards_mean) / rewards_std;
        else
            normalized_rewards = rewards - rewards_mean; % Just center
        end
        % Decide whether to use 'rewards' or 'normalized_rewards' below
        % Using normalized_rewards for critic training might be more stable
        rewards = normalized_rewards;
    end

    % --- Use the Critic for Baseline Estimate ---
    % Get encoder LSTM outputs [Hidden, Batch, Time] = [128, 64, 20]
    encLSTM_out = forward(actorNet, dlX, 'Outputs', 'encLSTM');
    % Average over Time dimension (dim 3) -> [128, 64, 1]
    encOutAvg = mean(encLSTM_out, 3);
    % Ensure format is CBT [C=hidden, B=batch, T=1] for Critic input layer
    encOutAvg = dlarray(extractdata(encOutAvg), 'CBT');

    baselineDL = forward(criticNet, encOutAvg); % Critic output [1, B, 1]
    baseline = squeeze(extractdata(baselineDL)); % -> [B] or [1, B]

    % --- Advantage Calculation ---
    adv = rewards - baseline(:); % Ensure column vector

    % Normalize advantage across the batch (zero mean, unit std dev)
    adv_mean = mean(adv);
    adv_std = std(adv, 1); % Use population std dev (flag=1) or sample std dev (flag=0)
    if adv_std > 1e-8 % Avoid division by zero or NaN
        adv = (adv - adv_mean) / adv_std;
    else
        adv = adv - adv_mean; % Just center if std dev is tiny
    end
    % --- End Normalization ---

    % --- Compute Losses and Gradients ---

    % Actor loss (includes entropy bonus)
    [gradA, lossA] = dlfeval(actorLoss_acc, actorNet, dlX, adv, tours, entropyCoefficient, temperature);
    % Critic loss
    [gradC, lossC] = dlfeval(criticLoss_acc, criticNet, encOutAvg, rewards);

    % --- Gradient Clipping ---
    gradA = thresholdL2Norm( gradA, gradThreshold);
    gradC = thresholdL2Norm( gradC, gradThreshold);

    % --- Update Networks using Adam ---
    [actorNet, avgGradA, avgSqGradA] = adamupdate(...
        actorNet, gradA, avgGradA, avgSqGradA, epoch, learnRateA);
    [criticNet, avgGradC, avgSqGradC] = adamupdate(...
        criticNet, gradC, avgGradC, avgSqGradC, epoch, learnRateC);

    % --- Store and Display Progress ---
    meanRewardEpoch = mean(rewards);
    historyReward(epoch) =  meanRewardEpoch;
    if mod(epoch, 20) == 0 || epoch == 1
        elapsedTime = toc;
        fprintf('Epoch %d/%d, ActLoss=%.4f, CritLoss=%.4f, AvgReward=%.3f (Best: %.3f, Patience: %d/%d), Time=%.2fs\n', ...
            epoch, numEpochs, double(gather(lossA)), double(gather(lossC)), ...
            meanRewardEpoch, bestReward, epochsSinceImprovement, patience, elapsedTime); % Added Best Reward & Patience
        tic; % Reset timer for next interval
    end

    % --- Early Stopping Check ---
    if meanRewardEpoch > bestReward + 1e-5 % Add small tolerance for floating point noise
        bestReward = meanRewardEpoch;
        epochsSinceImprovement = 0;
        % Save the state of the best model found so far
        bestActorNetState = actorNet.Learnables;
        bestCriticNetState = criticNet.Learnables;
    else
        epochsSinceImprovement = epochsSinceImprovement + 1;
    end

    if epochsSinceImprovement >= patience
        fprintf('\nEarly stopping triggered after %d epochs with no improvement.\n', patience);
        fprintf('Best average reward achieved: %.4f\n', bestReward);
        % Restore the best model state
        if ~isempty(bestActorNetState)
            actorNet.Learnables = bestActorNetState;
            criticNet.Learnables = bestCriticNetState;
            fprintf('Restored model state from epoch with best reward.\n');
        else
            fprintf('Warning: No improvement detected during training, keeping last model state.\n');
        end
        % Trim history if needed
        historyReward = historyReward(1:epoch);
        break; % Exit the training loop
    end
    % --- End Early Stopping Check ---

end % End Training Loop

% If loop finished without early stopping, ensure best model is loaded
if epoch == numEpochs && ~isempty(bestActorNetState) && epochsSinceImprovement > 0
    fprintf('\nTraining finished. Restoring model state from epoch with best reward (%.4f).\n', bestReward);
    actorNet.Learnables = bestActorNetState;
    criticNet.Learnables = bestCriticNetState;
elseif epoch == numEpochs
    fprintf('\nTraining finished.\n'); % No early stopping, already has last/best state
end

% Plot training rewards
figure;
% Determine the actual number of epochs run (in case of early stopping)
actualEpochs = length(historyReward);
plot(1:actualEpochs, historyReward(1:actualEpochs), '-o', 'MarkerSize', 4); % Plot only the epochs run

% Add a marker for the best reward epoch (if early stopping occurred)
if actualEpochs < numEpochs && ~isempty(bestActorNetState) % Check if early stopped and improvement occurred
    % Find the epoch where the best reward was achieved
    [~, bestEpochIdx] = max(historyReward(1:actualEpochs));
    hold on;
    plot(bestEpochIdx, bestReward, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r'); % Plot a red pentagram
    text(bestEpochIdx, bestReward, sprintf(' Best: %.3f (Epoch %d)', bestReward, bestEpochIdx), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r');
    hold off;
    title(sprintf('Training Progress (Stopped Early at Epoch %d)', actualEpochs));
else
    title('Training Progress');
end

xlabel('Epoch');
ylabel('Average Reward (Negative Length)');
grid on;
xlim([0, actualEpochs + 1]); % Adjust x-limit to actual epochs run

% (Optional) save trained actor and other data
save("NCO_trainedActor.mat", "actorNet", "criticNet", "historyReward");

%% 4) Greedy Inference with Monte Carlo Sampling
fprintf('Running greedy inference with the trained PtrNet actor...\n');

% Note: We can run the inference with a different number of locations!
nCities = 10;
coordsTest = rand(1, nCities, 2, 'single'); % Single instance B=1
% Permute to [C, B, T] -> [2, 1, 20]
dlXtest   = dlarray(permute(coordsTest,[3 1 2]), 'CBT');

% Monte Carlo parameters
numMCSamples = 200 ;
bestTourPred = [];
bestLenPred = Inf;

% Add temperature for inference
% Controls softmax sharpness (1.0=standard, <1.0=sharper, >1.0=smoother)
% <1.0 means more exploitation, >1.0 means more exploration
inferenceTemperature = 0.3; % Values < 1.0 favor exploitation
fprintf('Using inference temperature: %.2f\n', inferenceTemperature);

% Run multiple times and keep the best result
for i = 1:numMCSamples

    tourPred  = greedyDecode(actorNet, dlXtest, inferenceTemperature); % MODIFIED: Added temperature
    lenPred   = computeLengths(coordsTest, tourPred);       % RL length

    if lenPred < bestLenPred
        bestLenPred = lenPred;
        bestTourPred = tourPred;
    end

    % Optional: Show progress every 20 samples
    if mod(i, 20) == 0
        fprintf('  Monte Carlo samples: %d/%d, Current best: %.4f\n', i, numMCSamples, bestLenPred);
    end
end

% Use the best found tour
tourPred = bestTourPred;
lenPred = bestLenPred;

fprintf('RL Greedy tour length: %.4f\n', lenPred);
fprintf('RL Greedy tour sequence: %s\n', num2str(tourPred));

% Test with pre-trained model before RL fine-tuning
fprintf('Testing pre-trained model performance...\n');
tourPredPreTrain = greedyDecode(actorNet, dlXtest, inferenceTemperature);
lenPredPreTrain = computeLengths(coordsTest, tourPredPreTrain);
fprintf('Pre-trained model tour length: %.4f\n', lenPredPreTrain);

% Compare performance improvement from pre-training to RL
if lenPred < lenPredPreTrain
    improvementPercent = ((lenPredPreTrain - lenPred) / lenPredPreTrain) * 100;
    fprintf('RL fine-tuning improved tour length by %.2f%%\n', improvementPercent);
else
    fprintf('RL fine-tuning did not improve over pre-trained model\n');
end

%% --- Add Heuristic Comparison ---
fprintf('Running 2-opt heuristic...\n');
coordsTestMatrix = squeeze(double(coordsTest)); % Get Nx2 matrix for heuristic
rlStartCity = tourPred(1); % Get the first city from RL tour
[tourHeuristic, lenHeuristic] = solveTSP_2opt(coordsTestMatrix, rlStartCity);

fprintf('2-opt Heuristic tour length: %.4f\n', lenHeuristic);
fprintf('2-opt Heuristic tour sequence: %s\n', num2str(tourHeuristic));

% --- Calculate Optimality Gap (Relative to Heuristic) ---
% Note: lenHeuristic might not be the true optimum!
if lenHeuristic > 1e-9 % Avoid division by zero
    optimalityGap = ((lenPred - lenHeuristic) / lenHeuristic) * 100;
    fprintf('RL solution gap vs 2-opt: %.2f%%\n', optimalityGap);
else
    fprintf('Heuristic length is near zero, cannot calculate gap.\n');
end
% --- End Heuristic Comparison ---


%% 5) Visualize the Predicted Tour (and Heuristic Tour)
figure;
hold on;

% Extract coordinates and tour indices (ensure double for plotting/indexing)
coordsPlot = squeeze(double(coordsTest)); % Shape [N, 2]
tourPlotRL = double(squeeze(tourPred));      % Shape [N] (row vector)
tourPlotHeuristic = double(squeeze(tourHeuristic)); % Shape [N] (row vector)

% --- Plot Heuristic Tour ---
orderedCoordsHeuristic = coordsPlot(tourPlotHeuristic, :);
pathCoordsHeuristic = [orderedCoordsHeuristic; orderedCoordsHeuristic(1,:)];
plot(pathCoordsHeuristic(:,1), pathCoordsHeuristic(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('2-opt (L=%.3f)', lenHeuristic));

% --- Plot RL Tour ---
orderedCoordsRL = coordsPlot(tourPlotRL, :);
pathCoordsRL = [orderedCoordsRL; orderedCoordsRL(1,:)];
plot(pathCoordsRL(:,1), pathCoordsRL(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', sprintf('RL Greedy (L=%.3f)', lenPred));

% Plot the city locations as points
plot(coordsPlot(:,1), coordsPlot(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off'); % HandleVisibility off for legend

% Add labels (city numbers 1 to N)
for i = 1:nCities
    text(coordsPlot(i,1) + 0.01, coordsPlot(i,2) + 0.01, num2str(i), 'Color', 'k', 'FontSize', 10);
end

% Highlight the starting city (using city 1 as reference, or first in RL tour)
startCityIndex = tourPlotRL(1); % Start city from RL tour
plot(coordsPlot(startCityIndex, 1), coordsPlot(startCityIndex, 2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Start (RL)');

hold off;
title('RL vs 2-opt Heuristic TSP Tours');
xlabel('X Coordinate');
ylabel('Y Coordinate');
legend('show', 'Location', 'best');
axis equal;
grid on;
pad = 0.05;
xlim([min(coordsPlot(:,1))-pad, max(coordsPlot(:,1))+pad]);
ylim([min(coordsPlot(:,2))-pad, max(coordsPlot(:,2))+pad]);

%% Helper functions
function [tours, logProbs, entropy] = actorStep(net, dlX, temperature)
% --- Actor Step: Sample tours + log-probs ---
% dlX dims 'CBT' -> [C, B, T]

B = size(dlX,2); % Batch size
N = size(dlX,3); % Number of cities (Time steps)
% C_in = size(dlX, 1); % Input features (should be 2)

hiddenSize = net.Layers(strcmp({net.Layers.Name},'encLSTM')).NumHiddenUnits;

% 1. --- Encoder Pass ---
% encLSTM_out shape [Hidden, Batch, Time] = [128, 64, 20]
[encLSTM_out, ~] = forward(net, dlX, 'Outputs', 'encLSTM');

% --- Extract Projection Weights & Biases ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys (Project ONCE) ---
% Input encLSTM_out [128, 64, 20], Output keys [128, 64, 20] ('CBT' assumed by fullyconnect)
keys = fullyconnect(encLSTM_out, Wk, Bk);

% --- Initialize Decoder State ---
decH = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB'); % [128, 64]
decC = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB'); % [128, 64]

% --- Extract DECODER LSTM Weights ---
decLSTMLayer = net.Layers(strcmp({net.Layers.Name},'decLSTM_params'));
W_dec_lstm = decLSTMLayer.InputWeights;    % [512, 128]
R_dec_lstm = decLSTMLayer.RecurrentWeights;% [512, 128]
b_dec_lstm = decLSTMLayer.Bias;            % [512, 1]

% --- Initialize Decoder Loop Variables ---
mask = false(1, B, N); % Mask for visited cities [1, Batch, Time/Cities]
tours = zeros(B, N, 'uint32'); % Output tours [Batch, Cities]
logProbs = zeros(B, N, 'like', dlX); % Output logProbs [Batch, Cities]
entropy_steps = zeros(1, B, N, 'like', dlX); % Store step entropy [1, Batch, Cities]

% Initial decoder input: Zeros [Hidden, Batch, SeqLen=1] -> [128, 64, 1] ('CBT')
decIn = dlarray(zeros(hiddenSize, B, 1, 'like', dlX), 'CBT');

scaleFactor = sqrt(single(hiddenSize)); % For scaled dot-product

% 2. --- Decoder Loop ---
for t = 1:N
    % --- Decoder LSTM Step ---
    % Input decIn [128, B, 1], decH/decC [128, B] -> Output decOut [128, B, 1]
    [decOut, decH, decC] = lstm(decIn, decH, decC, W_dec_lstm, R_dec_lstm, b_dec_lstm);

    % --- Attention Mechanism (Scaled Dot-Product K-Q) ---
    % Project decoder output to Query
    % Input decOut [128, B, 1] -> query_input [128, B]
    query_input = reshape(decOut, hiddenSize, B);
    % query [128, B]
    query = fullyconnect(query_input, Wq, Bq, "DataFormat","CB");

    % Calculate Scores
    % keys [128, B, N=20]
    % query [128, B] -> reshape to [128, B, 1] for broadcasting
    reshaped_query = reshape(query, hiddenSize, B, 1);
    % scores = sum(keys .* reshaped_query, 1); -> [1, B, N=20]
    scores = sum(keys .* reshaped_query, 1);
    scores = scores / scaleFactor; % Scale

    % Add layer normalization (new)
    scores_mean = mean(scores, 3);
    scores_std = std(scores, 0, 3) + 1e-5;
    scores = (scores - scores_mean) ./ scores_std;

    % Apply mask
    scores(mask) = -inf;

    % --- Calculate Probabilities ---
    % Apply softmax over the Cities dimension (dim 3)
    % stripdims removes CBT label, DataFormat ensures softmax over N=20
    scores_ = stripdims(scores); % -> [1, B, N] numeric array
    scores_scaled = scores_ / temperature;
    probs = softmax(scores_scaled, "DataFormat", "TBC"); % Treat as T=1, B=B, C=N -> softmax over C=N -> [1, B, N]

    % --- Sample Next City ---
    idx = zeros(B, 1, 'uint32'); % Stores index (1 to N) for each batch item
    probs_gathered = gather(extractdata(probs)); % [1, B, N] -> standard array for sampling loop
    scores_gathered = gather(extractdata(scores)); % For fallback [1, B, N]
    mask_gathered = mask; % [1, B, N]

    for i = 1:B % Loop through batch items
        % Select probabilities for current batch item i: [1, 1, N] -> squeeze -> [1, N]
        current_probs_raw = squeeze(probs_gathered(1, i, :));
        p = max(0, current_probs_raw);
        p = p / (sum(p) + 1e-8); % Normalize

        if any(isnan(p)) || sum(p) < 1e-6 % Fallback
            current_mask = squeeze(mask_gathered(1, i, :)); % [N]
            valid_indices = find(~current_mask);
            if isempty(valid_indices)
                current_scores = squeeze(scores_gathered(1, i, :)); % [N]
                [~, max_idx_val] = max(current_scores);
                idx(i) = max_idx_val;
            else
                current_scores_valid = squeeze(scores_gathered(1, i, valid_indices));
                [~, max_idx_local] = max(current_scores_valid);
                idx(i) = valid_indices(max_idx_local);
            end
        else
            % Ensure p is row or column vector for randsample
            idx(i) = randsample(N, 1, true, p(:)); % p(:) ensures column vector
        end
    end % End batch loop

    % --- Store Results ---
    tours(:,t) = idx; % idx is [B, 1]

    % Calculate linear indices into probs [1, B, N] for the chosen cities idx [B, 1]
    % Need indices corresponding to (1, 1:B, idx)
    linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
    logProbs(:,t) = log(probs(linearIndicesProbs) + 1e-10); % logProbs is [B, N]

    % --- Calculate Entropy ---
    % Sum probs.*log(probs) over the Cities dimension (dim 3) -> [1, B, 1]
    stepEntropy = -sum(probs .* log(probs + 1e-10), 3);
    entropy_steps(1, :, t) = stepEntropy; % Store [1, B, 1] in [1, B, N] slice

    % --- Update Mask ---
    % Create linear indices for the mask [1, B, N]
    linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(idx));
    mask(linearIndicesMask) = true;

    % --- Prepare Input for Next Decoder Step ---
    % Select features from encLSTM_out [Hidden, B, N] using idx [B, 1]
    % Need features for (:, 1:B, idx)
    linearIndicesEnc = sub2ind([B, N], (1:B)', double(idx)); % Indices into dimensions 2 & 3
    featuresSelected = encLSTM_out(:, linearIndicesEnc); % -> [Hidden, B]
    % Reshape to [Hidden, B, SeqLen=1] for LSTM input and add 'CBT' label
    decIn = dlarray(reshape(featuresSelected, hiddenSize, B, 1), 'CBT');

end % End time step loop

% Sum entropy over tour steps (dimension 3) -> [1, B, 1] -> reshape -> [B, 1]
entropy = reshape(sum(entropy_steps, 3), B, 1);

end

function [tours, logProbs, entropy] = actorStep_coords(net, dlX, temperature, coordsBatch)
% --- Actor Step: Sample tours + log-probs ---
% coordsBatch: [B×N×2] is used to check crossings on partial tours

[~, B, N] = size(dlX);
% --- Initialize Decoder Loop Variables ---
mask = false(1, B, N); % Mask for visited cities [1, Batch, Time/Cities]
tours = zeros(B, N, 'uint32'); % Output tours [Batch, Cities]
logProbs = zeros(B, N, 'like', dlX); % Output logProbs [Batch, Cities]
hiddenSize = net.Layers(strcmp({net.Layers.Name},'encLSTM')).NumHiddenUnits;
% Initial decoder input: Zeros [Hidden, Batch, SeqLen=1] -> [128, 64, 1] ('CBT')
decIn    = dlarray(zeros(hiddenSize, B, 1), 'CBT');

% --- Extract Projection Weights & Biases ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Initialize Decoder State ---
decH = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB'); % [128, 64]
decC = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB'); % [128, 64]

% --- Extract DECODER LSTM Weights ---
decLSTMLayer = net.Layers(strcmp({net.Layers.Name},'decLSTM_params'));
W_dec_lstm = decLSTMLayer.InputWeights;    % [512, 128]
R_dec_lstm = decLSTMLayer.RecurrentWeights;% [512, 128]
b_dec_lstm = decLSTMLayer.Bias;            % [512, 1]

% 1. --- Encoder Pass ---
% encOut shape [Hidden, Batch, Time] = [128, 64, 20]
[encOut, ~] = forward(net, dlX, 'Outputs','encLSTM');
% --- Pre-calculate Keys (Project ONCE) ---
% Input encOut [128, 64, 20], Output keys [128, 64, 20] ('CBT' assumed by fullyconnect)
keys = fullyconnect(encOut, Wk, Bk);

scaleFactor = sqrt(single(hiddenSize)); % For scaled dot-product

% 2. --- Decoder Loop ---
for t = 1:N
    % --- Decoder LSTM Step ---
    % Standard LSTM + Query projection
    % Input decIn [128, B, 1], decH/decC [128, B] -> Output decOut [128, B, 1]
    [decOut, decH, decC] = lstm(decIn, decH, decC, W_dec_lstm, R_dec_lstm, b_dec_lstm);
    % --- Attention Mechanism (Scaled Dot-Product K-Q) ---
    % Project decoder output to Query
    % Input decOut [128, B, 1] -> query_input [128, B]
    query_input = reshape(decOut, hiddenSize, B);
    % query [128, B]
    query = fullyconnect(query_input, Wq, Bq, "DataFormat","CB");

    % Calculate Scores
    % keys [128, B, N=20]
    % query [128, B] -> reshape to [128, B, 1] for broadcasting
    reshaped_query = reshape(query, hiddenSize, B, 1);
    % scores = sum(keys .* reshaped_query, 1); -> [1, B, N=20]
    scores = sum(keys .* reshaped_query, 1);
    scores = scores / scaleFactor; % Scale

    % --- Crossing-based mask update ---
    % For each batch element, temporarily simulate choosing each candidate
    for b = 1:B
        partialTour = double(tours(b,1:t-1));  coords = squeeze(coordsBatch(b,:,:));
        for c = 1:N
            if ~mask(1,b,c)
                trialTour = [partialTour, c];
                if countCrossings(trialTour, coords) > 0
                    scores(1,b,c) = -inf;  % forbid crossing
                end
            end
        end
    end

    % Apply visited-city mask
    scores(mask) = -inf;

    % Softmax + sample
    probs = softmax(stripdims(scores)/temperature,"DataFormat","TBC");
    idx   = sampleFromProbs(probs);  % your existing sampling
    tours(:,t) = idx;
    logProbs(:,t) = log(probs(sub2ind(size(probs),ones(B,1),(1:B)',double(idx))) + 1e-10);

    % Update visited mask & decoder input for next step
    mask(sub2ind(size(mask),ones(B,1),(1:B)',double(idx))) = true;
    decIn = selectFeatures(encOut, idx);
end

entropy = computeEntropy(probs);
end

function tours = greedyDecode(net, dlX, temperature)
% --- Greedy Decoding (Upgraded Attention) ---
% dlX dims 'CBT' -> [C, B, T] - NOTE: B=1 for inference here
% temperature - Controls softmax sharpness (added parameter)

% Set default temperature if not provided
if nargin < 3
    temperature = 1.0; % Default temperature
end

B = size(dlX,2); % Batch size (Expected to be 1)
N = size(dlX,3); % Number of cities (Time steps)
% C_in = size(dlX, 1);

hiddenSize = net.Layers(strcmp({net.Layers.Name},'encLSTM')).NumHiddenUnits;

% 1. --- Encoder Pass ---
% encLSTM_out shape [Hidden, B, Time] = [128, 1, 20]
[encLSTM_out, ~] = forward(net, dlX, 'Outputs', 'encLSTM');

% --- Extract Projection Weights & Biases ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys ---
% Input encLSTM_out [128, 1, 20], Output keys [128, 1, 20]
keys = fullyconnect(encLSTM_out, Wk, Bk);

% --- Initialize Decoder State ---
decH = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB'); % [128, 1]
decC = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB'); % [128, 1]

% --- Extract DECODER LSTM Weights ---
decLSTMLayer = net.Layers(strcmp({net.Layers.Name},'decLSTM_params'));
W_dec_lstm = decLSTMLayer.InputWeights;
R_dec_lstm = decLSTMLayer.RecurrentWeights;
b_dec_lstm = decLSTMLayer.Bias;

% --- Initialize Decoder Loop Variables ---
mask = false(1, B, N); % [1, 1, 20]
tours = zeros(B, N, 'uint32'); % [1, 20]
% Initial decoder input [128, 1, 1] ('CBT')
decIn = dlarray(zeros(hiddenSize, B, 1, 'like', dlX), 'CBT');

scaleFactor = sqrt(single(hiddenSize));

% 2. --- Decoder Loop ---
for t = 1:N
    % --- Decoder LSTM Step ---
    % Input decIn [128, 1, 1], decH/decC [128, 1] -> Output decOut [128, 1, 1]
    [decOut, decH, decC] = lstm(decIn, decH, decC, W_dec_lstm, R_dec_lstm, b_dec_lstm);

    % --- Attention Mechanism ---
    % Project Query
    query_input = reshape(decOut, hiddenSize, B); % [128, 1]
    query = fullyconnect(query_input, Wq, Bq, "DataFormat","CB"); % [128, 1]

    % Calculate Scores
    % keys [128, 1, 20]
    % query [128, 1] -> reshape to [128, 1, 1]
    reshaped_query = reshape(query, hiddenSize, B, 1);
    % scores = sum(keys .* reshaped_query, 1); -> [1, 1, 20]
    scores = sum(keys .* reshaped_query, 1);
    scores = scores / scaleFactor; % Scale

    % Add layer normalization (new)
    scores_mean = mean(scores, 3);
    scores_std = std(scores, 0, 3) + 1e-5;
    scores = (scores - scores_mean) ./ scores_std;

    % Apply mask
    scores(mask) = -inf;

    % --- Select Next City (Greedy) ---
    % If temperature < 1, we'll sample with the modified temperature
    if temperature < 0.99 && t < N  % For all steps except last, use temperature sampling
        % Apply temperature scaling
        scores_ = stripdims(scores);
        scores_scaled = scores_ / temperature;
        probs = softmax(scores_scaled, "DataFormat", "TBC"); % [1, B, N]

        % Convert to standard array for sampling
        probs_gathered = gather(extractdata(probs)); % [1, B, N]
        p = squeeze(probs_gathered(1, 1, :)); % [N,1]
        p = max(0, p);
        p = p / (sum(p) + 1e-8); % Normalize

        % Sample according to temperature-adjusted distribution
        idx = randsample(N, 1, true, p);
        idx = uint32(idx);
    else
        % Use greedy (argmax) for last step or if temperature=1
        [~, idx] = max(scores, [], 3); % Find max index along dim 3
        idx = squeeze(idx); % Remove singleton dims -> scalar index
        idx = uint32(extractdata(idx));
    end

    % --- Store Results and Update State ---
    tours(:,t) = idx; % Store scalar idx in tours [1, 20]

    % --- Update Mask ---
    % Need index for mask [1, 1, 20] corresponding to (1, 1, idx)
    linearIndicesMask = sub2ind(size(mask), 1, 1, double(idx));
    mask(linearIndicesMask) = true;

    % --- Prepare Input for Next Decoder Step ---
    % Select features from encLSTM_out [128, 1, 20] using idx
    % Need features for (:, 1, idx)
    linearIndicesEnc = sub2ind([B, N], 1, double(idx)); % Index into dimensions 2 & 3
    featuresSelected = encLSTM_out(:, linearIndicesEnc); % -> [128, 1]
    % Reshape to [128, 1, 1] ('CBT')
    decIn = dlarray(reshape(featuresSelected, hiddenSize, B, 1), 'CBT');

end % End time step loop
end

function [gradients, loss] = actorLoss(net, dlX, adv, tours, entropyCoeff, temperature)
% --- Loss Functions ---
% Re-calculate necessary outputs within dlfeval context for gradient tracking

[C, B, nCities] = size(dlX); %#ok<ASGLU>

[logProbs, entropy] = calculateLogProbsAndEntropyForTours(net, dlX, tours, temperature);

% Policy Gradient Term
sumLogProbs = sum(logProbs, 2); % Shape: [B, 1]
adv = dlarray(adv(:)); % Ensure adv is dlarray column vector [B, 1]
policyLoss = -mean(sumLogProbs .* adv);

% Entropy Bonus Term (maximize entropy -> minimize negative entropy)
% entropy shape should be [B, 1]
entropyLoss = -mean(entropy) ; %/ nCities; % Average entropy over batch

% Combined Loss
loss = policyLoss + entropyCoeff * entropyLoss;

gradients = dlgradient(loss, net.Learnables);
end

function [logProbs, entropy] = calculateLogProbsAndEntropyForTours(net, dlX, tours, temperature)
% Helper to calculate logProbs AND entropy for existing tours inside dlfeval
% This function needs to mirror actorStep exactly,
% except it USES the provided 'tours' instead of sampling 'idx'.

% dlX dims 'CBT' -> [C, B, T]
B = size(dlX,2); % Batch size
N = size(dlX,3); % Number of cities (Time steps)
hiddenSize = net.Layers(strcmp({net.Layers.Name},'encLSTM')).NumHiddenUnits;

% 1. --- Encoder Pass ---
[encLSTM_out, ~] = forward(net, dlX, 'Outputs', 'encLSTM');

% --- Extract Projection Weights & Biases ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys ---
keys = fullyconnect(encLSTM_out, Wk, Bk);

% --- Initialize Decoder State ---
decH = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB');
decC = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB');

% --- Extract DECODER LSTM Weights ---
decLSTMLayer = net.Layers(strcmp({net.Layers.Name},'decLSTM_params'));
W_dec_lstm = decLSTMLayer.InputWeights;
R_dec_lstm = decLSTMLayer.RecurrentWeights;
b_dec_lstm = decLSTMLayer.Bias;

% --- Initialize Decoder Loop Variables ---
mask = false(1, B, N); % [1, B, N]
logProbs = zeros(B, N, 'like', dlX); % [B, N] - Target shape
entropy_steps = zeros(1, B, N, 'like', dlX); % [1, B, N]
decIn = dlarray(zeros(hiddenSize, B, 1, 'like', dlX), 'CBT'); % [128, B, 1]

scaleFactor = sqrt(single(hiddenSize));

% 2. --- Decoder Loop ---
for t = 1:N
    % --- Decoder LSTM Step ---
    [decOut, decH, decC] = lstm(decIn, decH, decC, W_dec_lstm, R_dec_lstm, b_dec_lstm);

    % --- Attention Mechanism ---
    query_input = reshape(decOut, hiddenSize, B);
    query = fullyconnect(query_input, Wq, Bq, "DataFormat","CB");
    reshaped_query = reshape(query, hiddenSize, B, 1);
    scores = sum(keys .* reshaped_query, 1); % -> [1, B, N]
    scores = scores / scaleFactor;
    scores(mask) = -inf;

    % --- Calculate Probabilities ---
    scores_ = stripdims(scores);
    scores_scaled = scores_ / temperature;
    probs = softmax(scores_scaled, "DataFormat", "TBC"); % Softmax over N -> [1, B, N]

    % --- Use provided tour index ---
    idx = tours(:, t); % idx is [B, 1], contains indices from 1 to N

    % --- Store Log Probability of the chosen action ---
    % Calculate linear indices into probs [1, B, N] for the chosen cities idx [B, 1]
    linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
    % Assign to logProbs [B, N]
    logProbs(:,t) = log(probs(linearIndicesProbs) + 1e-10); % Result is [B, 1] -> OK

    % --- Calculate Entropy ---
    stepEntropy = -sum(probs .* log(probs + 1e-10), 3); % -> [1, B, 1]
    entropy_steps(1, :, t) = stepEntropy; % Store in [1, B, N]

    % --- Update Mask using provided tour index ---
    linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(idx));
    mask(linearIndicesMask) = true;

    % --- Prepare Input for Next Decoder Step using provided tour index ---
    linearIndicesEnc = sub2ind([B, N], (1:B)', double(idx)); % Indices into dimensions 2 & 3
    featuresSelected = encLSTM_out(:, linearIndicesEnc); % -> [Hidden, B]
    decIn = dlarray(reshape(featuresSelected, hiddenSize, B, 1), 'CBT'); % -> [Hidden, B, 1]

end % End time step loop

% Sum entropy over tour steps (dimension 3) -> [1, B, 1] -> reshape -> [B, 1]
entropy = reshape(sum(entropy_steps, 3), B, 1);
end

function [gradients, loss] = criticLoss(net, encOutAvg, rewards)
% Critic Loss (remains the same, but check input dims)
% encOutAvg should be [Hidden, Batch, 1] 'CBT'
% Critic Input Layer expects [Hidden, 1, Batch] 'CBT' - Needs correction in main loop

% Assuming encOutAvg IS correctly formatted [Hidden, 1, Batch] 'CBT' by main loop
baselinePredicted = forward(net, encOutAvg); % Output [1, 1, B] 'CBT'

baselinePredicted = squeeze(baselinePredicted); % -> [1, B] or [B]
rewards = dlarray(rewards(:)); % Ensure [B, 1]
baselinePredicted = baselinePredicted(:); % Ensure [B, 1]

loss = mean((baselinePredicted - rewards).^2);
gradients = dlgradient(loss, net.Learnables);
end

function clippedGradients = thresholdL2Norm(gradientsTable, threshold)
%thresholdL2Norm Clips gradients in a table format based on global L2 norm.
%
%   clippedGradients = thresholdL2Norm(gradientsTable, threshold)
%   calculates the L2 norm across all gradient values in the 'Value'
%   column of the input gradientsTable. If the norm exceeds the threshold,
%   all gradients are scaled down proportionally.
%
%   Inputs:
%       gradientsTable - A table (like the output of dlgradient) where
%                        each row represents a learnable parameter and the
%                        'Value' column contains a cell with the dlarray gradient.
%       threshold      - The maximum allowed L2 norm for the gradients.
%
%   Outputs:
%       clippedGradients - A table with the same structure as gradientsTable,
%                          but with gradient values potentially scaled.

% Calculate L2 norm across all gradients in the table.
gradNorm = single(0); % Use single if gradients are single
numGrads = height(gradientsTable);
gradientValues = cell(numGrads, 1); % Store original dlarrays temporarily

for i = 1:numGrads
    currentGrad = gradientsTable.Value{i}; % Get the dlarray gradient
    if ~isempty(currentGrad) && isa(currentGrad, 'dlarray')
        gradientValues{i} = currentGrad; % Store the dlarray
        % Accumulate squared norm (ensure type consistency)
        gradNorm = gradNorm + sum(currentGrad(:).^2);
    end
end
gradNorm = sqrt(gradNorm);

% Create a copy of the input table to modify. This is crucial as
% dlupdate expects a modified version of the input structure/table.
clippedGradients = gradientsTable;

% Apply clipping if norm exceeds threshold
if gradNorm > threshold
    normScale = threshold / gradNorm;
    % fprintf('Clipping Gradients: Norm=%.4f, Scale=%.4f\n', gradNorm, normScale); % Optional debug

    % Iterate again to apply scaling
    for i = 1:numGrads
        if ~isempty(gradientValues{i}) % Check if we stored a gradient for this row
            % Scale the original gradient and update the 'Value' cell in the copied table
            clippedGradients.Value{i} = gradientValues{i} * normScale;
        end
    end
    % else % Optional debug
    % fprintf('Gradients Norm=%.4f (No Clipping)\n', gradNorm);
end
end

function L = computeLengths(coords, tours)
% --- Compute tour lengths  ---

B = size(coords,1);
% N = size(coords,2);
if isa(coords, 'dlarray')
    coords = extractdata(coords);
end
if ~isa(coords, 'single') && ~isa(coords, 'double')
    coords = double(coords);
end
if ~isa(tours, 'double')
    tours = double(tours);
end

L = zeros(B,1,'like',coords);

for i=1:B
    tour_indices = squeeze(tours(i,:));
    city_coords = squeeze(coords(i,:,:));
    ordered_coords = city_coords(tour_indices, :);
    rolled_coords = circshift(ordered_coords, -1, 1);
    diffs = rolled_coords - ordered_coords;
    segment_lengths = sqrt(sum(diffs.^2, 2));
    L(i) = sum(segment_lengths);
end

end

function [bestTour, bestLength] = solveTSP_2opt(coords, startCity)
%solveTSP_2opt Solves TSP approximately using the 2-opt heuristic method.
%
%   [bestTour, bestLength] = solveTSP_2opt(coords)
%   Inputs:
%       coords - Nx2 matrix of city coordinates.
%   Outputs:
%       bestTour - Row vector representing the order of city indices (1 to N).
%       bestLength - The length of the best tour found.

% Add startCity parameter (optional with default of 1)
if nargin < 2
    startCity = 1; % Default start city
end

n = size(coords, 1);
if n <= 3
    % For trivial cases, ensure we start with startCity
    bestTour = [startCity, setdiff(1:n, startCity)];
    bestLength = calculateTourLength(coords, bestTour);
    return;
end

% Generate initial tour starting from startCity
currentTour = nearestNeighborTour(coords, startCity);
currentLength = calculateTourLength(coords, currentTour);

bestTour = currentTour;
bestLength = currentLength;
improved = true;

% --- 2. Iteratively Apply 2-opt Swaps ---
while improved
    improved = false;
    for i = 1 : n-1 % First edge starting node index
        for k = i+1 : n % Second edge starting node index
            % Consider swapping edge (i, i+1) and (k, k+1 mod n)
            % New tour segment: reverse path from i+1 to k
            newTour = twoOptSwap(currentTour, i, k);
            newLength = calculateTourLength(coords, newTour);

            if newLength < currentLength - 1e-6 % Use tolerance for floating point
                currentTour = newTour;
                currentLength = newLength;
                improved = true;
                % Update best found if current is better
                if currentLength < bestLength
                    bestLength = currentLength;
                    bestTour = currentTour;
                end
                % Restart inner loops if improvement found (can be faster)
                % break; % Break inner 'k' loop
            end
        end % end 'k' loop
        % if improved % Break outer 'i' loop if restart strategy used
        %     break;
        % end
    end % end 'i' loop
    % If using restart strategy, the while loop continues if improved was true
end % end while loop

end

function tourLength = calculateTourLength(coords, tour)
% --- Calculate tour length ---
% n = length(tour);
orderedCoords = coords(tour,:);
pathCoords = [orderedCoords; orderedCoords(1,:)]; % Close the loop
diffs = diff(pathCoords, 1, 1); % Differences between consecutive points
segmentLengths = sqrt(sum(diffs.^2, 2));
tourLength = sum(segmentLengths);
end

function newTour = twoOptSwap(tour, i, k)
% --- Perform 2-opt swap ---
% Reverses the path segment between index i+1 and k (inclusive).
% In each 2-opt iteration, you remove two existing edges and reconnect the
% tour by “swapping” them—equivalently, by reversing the order of the nodes
% between two chosen positions. This simple operation systematically
% eliminates crossing edges and often yields shorter tours

newTour = tour; % Start with original
% Indices need careful handling if using 1-based indexing
% Segment to reverse is from index i+1 to k
segmentToReverse = newTour(i+1 : k);
newTour(i+1 : k) = fliplr(segmentToReverse); % Reverse the segment
end

function tour = nearestNeighborTour(coords, startCity)
% --- Nearest Neighbor Initial Tour ---
if nargin < 2
    startCity = 1; % Default
end

n = size(coords, 1);
tour = zeros(1, n);
visited = false(1, n);
currentCity = startCity; % Start at specified city
tour(1) = currentCity;
visited(currentCity) = true;

for i = 2:n
    minDist = inf;
    nearestCity = -1;
    for j = 1:n
        if ~visited(j)
            dist = norm(coords(currentCity,:) - coords(j,:));
            if dist < minDist
                minDist = dist;
                nearestCity = j;
            end
        end
    end
    currentCity = nearestCity;
    tour(i) = currentCity;
    visited(currentCity) = true;
end
end

function [gradients, loss] = supervisedLoss(net, dlX, targetTours, teacherForcingRatio)
% Supervised loss function with teacher forcing for the pointer network
% dlX: [C, B, T] input coordinates
% targetTours: [B, T] target tours from 2-opt

B = size(dlX, 2);          % Batch size
N = size(dlX, 3);          % Number of cities
hiddenSize = net.Layers(strcmp({net.Layers.Name}, 'encLSTM')).NumHiddenUnits;

% --- Encoder Pass ---
[encLSTM_out, ~] = forward(net, dlX, 'Outputs', 'encLSTM');

% --- Extract Projection Weights & Biases ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys ---
keys = fullyconnect(encLSTM_out, Wk, Bk);

% --- Initialize Decoder State ---
decH = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB');
decC = dlarray(zeros(hiddenSize, B, 'like', dlX), 'CB');

% --- Extract DECODER LSTM Weights ---
decLSTMLayer = net.Layers(strcmp({net.Layers.Name}, 'decLSTM_params'));
W_dec_lstm = decLSTMLayer.InputWeights;
R_dec_lstm = decLSTMLayer.RecurrentWeights;
b_dec_lstm = decLSTMLayer.Bias;

% --- Initialize Decoder Loop Variables ---
mask = false(1, B, N);
logProbs = zeros(B, N, 'like', dlX);
decIn = dlarray(zeros(hiddenSize, B, 1, 'like', dlX), 'CBT');

scaleFactor = sqrt(single(hiddenSize));

% --- Decoder Loop with Teacher Forcing ---
for t = 1:N
    % --- Decoder LSTM Step ---
    [decOut, decH, decC] = lstm(decIn, decH, decC, W_dec_lstm, R_dec_lstm, b_dec_lstm);

    % --- Attention Mechanism ---
    query_input = reshape(decOut, hiddenSize, B);
    query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB");
    reshaped_query = reshape(query, hiddenSize, B, 1);
    scores = sum(keys .* reshaped_query, 1);
    scores = scores / scaleFactor;

    % Apply mask for already visited cities
    scores(mask) = -inf;

    % Layer normalization for attention scores - FIXED VERSION
    scores_data = extractdata(scores);  % Extract numeric data
    scores_data(isinf(scores_data)) = NaN;  % Convert -inf to NaN for mean calculation
    scores_mean = mean(scores_data, 3, 'omitnan');  % Use omitnan instead
    scores_std = std(scores_data, 0, 3, 'omitnan') + 1e-5;

    % Convert back to dlarray and apply normalization
    scores_mean = dlarray(scores_mean);
    scores_std = dlarray(scores_std);
    scores = (scores - scores_mean) ./ scores_std;

    % Ensure masked cities stay masked after normalization
    scores(mask) = -inf;

    % --- Calculate Probabilities ---
    scores_ = stripdims(scores);
    probs = softmax(scores_, "DataFormat", "TBC");

    % --- Get target indices from 2-opt tours ---
    idx = targetTours(:, t);

    % --- Calculate log probabilities for loss ---
    linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
    logProbs(:, t) = log(probs(linearIndicesProbs) + 1e-10);

    % --- Update mask using target indices ---
    linearIndicesMask = sub2ind(size(mask), ones(B, 1), (1:B)', double(idx));
    mask(linearIndicesMask) = true;

    % --- Teacher forcing: mix between using target and using prediction ---
    if t < N  % For all but the last step
        useTeacherForcing = rand() < teacherForcingRatio;

        if useTeacherForcing
            % Teacher forcing: use target index
            nextIndices = double(idx);
        else
            % Sample from policy for next step (model prediction)
            probs_gathered = gather(extractdata(probs));
            nextIndices = zeros(B, 1, 'double');

            for i = 1:B
                p = squeeze(probs_gathered(1, i, :));
                p = max(0, p) / sum(max(0, p));
                nextIndices(i) = randsample(N, 1, true, p);
            end
        end

        % Select features for next input
        linearIndicesEnc = sub2ind([B, N], (1:B)', nextIndices);
        featuresSelected = encLSTM_out(:, linearIndicesEnc);
        decIn = dlarray(reshape(featuresSelected, hiddenSize, B, 1), 'CBT');
    end
end

% Cross-entropy loss (negative log likelihood of target tours)
loss = -mean(sum(logProbs, 2));
gradients = dlgradient(loss, net.Learnables);
end

function n = countCrossings(tour, coords)
%COUNTCROSSINGS  Returns number of intersecting edge pairs in a TSP tour
% tour: [1×N] permutation of cities
% coords: [N×2] city coordinates

N = numel(tour);
n = 0;
% Precompute city positions
pts = coords(tour, :);
for i = 1:N-1
    A = pts(i, :);    B = pts(i+1, :);
    for j = i+2:N-1
        C = pts(j, :);  D = pts(j+1, :);
        if segmentsIntersect(A,B,C,D)
            n = n + 1;
        end
    end
end
end

function tf = segmentsIntersect(A,B,C,D)
% Line segment intersection test
% Check if segments AB and CD intersect (excluding shared endpoints)
% Using orient tests and bounding boxes
    function o = orient(P,Q,R)
        o = (Q(1)-P(1))*(R(2)-P(2)) - (Q(2)-P(2))*(R(1)-P(1));
    end
o1 = orient(A,B,C);
o2 = orient(A,B,D);
o3 = orient(C,D,A);
o4 = orient(C,D,B);
tf = (o1*o2 < 0) && (o3*o4 < 0);
end

function idx = sampleFromProbs(probs)
% sampleFromProbs  Sample one index per batch element given a softmax tensor
%   probs : dlarray or numeric array of shape [1, B, N] where
%           probs(1,b,:) are nonnegative and sum to 1 over N
%   idx   : [B×1] uint32 vector of sampled city indices

% Convert to plain numeric array if needed
if isa(probs, 'dlarray')
    p = gather(extractdata(probs));
else
    p = probs;
end

% Dimensions
[~, B, N] = size(p);
idx = zeros(B, 1, 'uint32');

% For each batch element, sample from its categorical distribution
for b = 1:B
    pb = squeeze(p(1, b, :));
    pb = max(pb, 0);               % clip negatives (just in case)
    if sum(pb) == 0
        pb(:) = 1/N;              % uniform fallback
    else
        pb = pb / sum(pb);        % normalize
    end
    idx(b) = uint32( randsample(N, 1, true, pb) );
end
end

function decIn = selectFeatures(encOut, idx)
% selectFeatures  Extracts the encoder feature for each batch at the chosen index
%   encOut : dlarray of size [hiddenSize, B, N] ('CBT')
%   idx    : [B×1] uint32 vector of city indices (1…N)
%   decIn  : dlarray of size [hiddenSize, B, 1] ('CBT') ready for next LSTM step

[hiddenSize, B, ~] = size(encOut);
% Preallocate numeric array of the right type
data = zeros(hiddenSize, B, 1, 'like', encOut);

% Pull out the feature vector at encOut(:, b, idx(b)) for each sample
for b = 1:B
    data(:, b, 1) = encOut(:, b, idx(b));
end

% Return as dlarray with the correct label
decIn = dlarray(data, 'CBT');
end

function entropy = computeEntropy(probs)
% computeEntropy  Computes -∑ p·log(p) over the city dimension for each batch
%   probs   : dlarray or numeric array of shape [1, B, N] with probabilities
%   entropy : dlarray [B×1] of entropies

% Extract numeric data if needed
if isa(probs, 'dlarray')
    p = gather(extractdata(probs));
else
    p = probs;
end

[~, B, N] = size(p);
e = zeros(B,1,'like',p);

for b = 1:B
    pb = squeeze(p(1,b,:));
    % ensure numerical stability
    pb(pb < 1e-12) = [];
    e(b) = -sum(pb .* log(pb));
end

% Return as dlarray for compatibility in training loop
entropy = dlarray(e);
end
