%% Neural Combinatorial Optimization (NCO) example, using a GNN model
% # Traveling Salesman Problem (TSP) Solver Using NCO, with Reinforcement Learning
%
% ## Workflow Overview (GNN Based)
%
% 1. **Initialization and Setup**
%    - Define hyperparameters (locations count, embedding size, batch size)
%    - Set learning rates and entropy coefficients
%    - Configure early stopping parameters
%
% 2. **Network Architecture**
%    - Define actor network with GNN encoder and attention mechanism
%    - Define critic network for value estimation (using pooled GNN output)
%    - Initialize both networks
%
% 3. **Pre-training Phase (Optional)**
%    - Generate TSP instances and solve with 2-opt
%    - Train Actor network via supervised learning to mimic 2-opt steps
%
% 4. **Reinforcement Learning Training Loop**
%    - Generate random TSP instances
%    - Run GNN-based actor for tour sampling (using attention over node embeddings)
%    - Compute tour lengths and corresponding rewards
%    - Use critic to estimate baseline values from pooled GNN embeddings
%    - Calculate advantage (reward - baseline)
%    - Compute actor and critic losses
%    - Apply gradient clipping and parameter updates
%    - Track performance and apply early stopping
%
% 5. **Inference and Evaluation**
%    - Perform greedy/sampling decoding using GNN-based actor
%    - Compare with 2-opt heuristic solution
%    - Calculate optimality gap
%    - Visualize tours for comparison
%
% 6. **Helper Functions**
%    - GNN Forward Pass (`gnnForward`)
%    - Tour length calculation (`computeLengths`)
%    - Gradient clipping (`thresholdL2Norm`)
%    - Actor and critic loss functions (`actorLoss_GNN`, `criticLoss`)
%    - GNN-based step functions (`actorStep_GNN`, `greedyDecode_GNN`, `calculateLogProbsAndEntropyForTours_GNN`)
%    - 2-opt local search implementation (`solveTSP_2opt`, helpers)
%

% Copyright 2025, The MathWorks, Inc.

%% Hyperparameters
nCities    = 10;        % Number of cities
nodeEmbSize = 32;       % GNN node embedding dimension (Match original hiddenSize)
numGNNLayers = 3;       % Number of GNN layers to stack
batchSize  = 512;       % Minibatch size
numEpochs  = 1000;      % RL training epochs
gradThreshold = 2.0;    % L2 norm threshold for gradient clipping

% Learning Rates
learnRateA = 5e-4; % Actor LR (Adjusted for GNN)
learnRateC = 1e-3; % Critic LR (Adjusted for GNN)
learnRatePre = 1e-3; % Learning rate for pre-training

% Entropy Coefficient
entropyCoefficient = 0.01; % Coefficient for entropy bonus in actor loss

% Temperature parameter
temperature = 0.8; % For training sampling
inferenceTemperature = 0.4; % For final inference sampling/greedy

% Pre-training Parameters
numPretrainEpochs = 50; % Number of supervised pre-training epochs
pretrainBatchSize = 128; % Batch size for pre-training

% Early Stopping Parameters (for RL phase)
patiencePercentage = 0.30;
patience = ceil(numEpochs * patiencePercentage);

% Optional: For reproducibility
rng(1);

%% 1) Define Pointer (Actor) Network (GNN based)
actorLG = layerGraph();

% Input Layer
inputLayer = sequenceInputLayer(2, 'Name', 'in', 'Normalization', 'none');
actorLG = addLayers(actorLG, inputLayer);

% Initial Node Embedding
nodeEmbFCLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'nodeEmbFC');
nodeEmbReluLayer = reluLayer('Name', 'nodeEmbRelu');
actorLG = addLayers(actorLG, nodeEmbFCLayer);
actorLG = addLayers(actorLG, nodeEmbReluLayer);
actorLG = connectLayers(actorLG, 'in', 'nodeEmbFC');
actorLG = connectLayers(actorLG, 'nodeEmbFC', 'nodeEmbRelu');

% Define AND Connect GNN Layers sequentially for graph validity
lastGNNInputLayerName = 'nodeEmbRelu';
lastGNNOutputLayerName = '';
gnnLayerNames = strings(1, numGNNLayers);
for i = 1:numGNNLayers
    gnnLayerName = ['gnnFC' num2str(i)];
    gnnLayerNames(i) = gnnLayerName;
    gnnFCLayer = fullyConnectedLayer(nodeEmbSize, 'Name', gnnLayerName);
    actorLG = addLayers(actorLG, gnnFCLayer);
    actorLG = connectLayers(actorLG, lastGNNInputLayerName, gnnLayerName);
    lastGNNInputLayerName = gnnLayerName;
    lastGNNOutputLayerName = gnnLayerName;
end

% Define Attention Projection Layers
fcKeyLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'fcKey');
fcValueLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'fcValue');
fcQueryLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'fcQuery');
actorLG = addLayers(actorLG, fcKeyLayer);
actorLG = addLayers(actorLG, fcValueLayer);
actorLG = addLayers(actorLG, fcQueryLayer);

% Connect Attention Layers to satisfy graph requirements
if ~isempty(lastGNNOutputLayerName)
    actorLG = connectLayers(actorLG, lastGNNOutputLayerName, 'fcKey');
    actorLG = connectLayers(actorLG, lastGNNOutputLayerName, 'fcValue');
    actorLG = connectLayers(actorLG, lastGNNOutputLayerName, 'fcQuery');
else % Connect from embedding if no GNN layers
    actorLG = connectLayers(actorLG, 'nodeEmbRelu', 'fcKey');
    actorLG = connectLayers(actorLG, 'nodeEmbRelu', 'fcValue');
    actorLG = connectLayers(actorLG, 'nodeEmbRelu', 'fcQuery');
end

actorNet = dlnetwork(actorLG);

% Initialize Actor
dummyX_GNN = dlarray(rand(2, pretrainBatchSize, nCities, 'single'), 'CBT'); % Use pretrain batch size
actorNet = initialize(actorNet, dummyX_GNN);
fprintf('Actor network initialized (GNN based).\n');


%% 2) Define Critic Network (Takes Pooled GNN Output)
layersCritic = [
    % Input is the POOLED GNN embeddings [Emb, 1, Batch] 'CBT'
    sequenceInputLayer(nodeEmbSize,'Name','cIn', 'Normalization','none')
    fullyConnectedLayer(nodeEmbSize*2,'Name','cFC1');
    reluLayer('Name','cRelu1')
    fullyConnectedLayer(nodeEmbSize,'Name','cFC2');
    reluLayer('Name','cRelu2')
    fullyConnectedLayer(1,'Name','cOut')
    ];
criticNet = dlnetwork(layersCritic);

% Initialize critic - Input needs format [Emb, 1, Batch] 'CBT'
dummyCriticInPooled = dlarray(rand(nodeEmbSize, 1, batchSize, 'single'), 'CBT');
criticNet = initialize(criticNet, dummyCriticInPooled);
fprintf('Critic network initialized (takes pooled GNN output).\n');

%% 3) Pre-training Phase using 2-Opt Solutions
fprintf('\n--- Starting Pre-training Phase (%d epochs) ---\n', numPretrainEpochs);
avgGradPre = []; avgSqGradPre = [];
tic;

for preEpoch = 1:numPretrainEpochs
    % Generate coordinates for pre-training batch
    coordsPre = rand(pretrainBatchSize, nCities, 2, 'single');
    dlXPre = dlarray(permute(coordsPre,[3 1 2]), 'CBT'); % [C, B, T]

    % Generate target tours using 2-opt for each instance in the batch
    targetTours = zeros(pretrainBatchSize, nCities, 'uint32');
    for i = 1:pretrainBatchSize
        coordsInstance = squeeze(coordsPre(i,:,:));
        targetTours(i,:) = solveTSP_2opt(coordsInstance); % Start city default 1
    end

    % Perform supervised learning update using dlfeval
    [gradsPre, lossPre] = dlfeval(@pretrainLoss, actorNet, dlXPre, targetTours, numGNNLayers, nCities, nodeEmbSize);

    % Clip gradients (optional but recommended)
    gradsPre = thresholdL2Norm( gradsPre, gradThreshold);

    % Update Actor network using Adam
    [actorNet, avgGradPre, avgSqGradPre] = adamupdate(...
        actorNet, gradsPre, avgGradPre, avgSqGradPre, preEpoch, learnRatePre);

    if mod(preEpoch, 10) == 0 || preEpoch == 1
        fprintf(' Pretrain Epoch %d/%d, Loss: %.4f, Time: %.2fs\n', ...
                preEpoch, numPretrainEpochs, gather(extractdata(lossPre)), toc);
        tic;
    end
end
fprintf('--- Pre-training Phase Finished ---\n\n');


%% 4) Reinforcement Learning Training Loop
fprintf('--- Starting Reinforcement Learning Phase (%d epochs) ---\n', numEpochs);
avgGradA = []; avgSqGradA = []; % Reset Adam states for RL phase
avgGradC = []; avgSqGradC = [];

% --- Early Stopping Parameters ---
bestReward = -inf;
epochsSinceImprovement = 0;
bestActorNetState = [];
bestCriticNetState = [];
% --------------------------------

fprintf('RL Training (nCities=%d, nodeEmbSize=%d, batchSize=%d, gradClip=%.1f, entropyCoeff=%.3f)...\n', ...
    nCities, nodeEmbSize, batchSize, gradThreshold, entropyCoefficient);
fprintf('Early stopping patience: %d epochs\n', patience);
tic;

historyReward = zeros(1,numEpochs);

for epoch = 1:numEpochs
    % Generate random TSP batch [B, N, C=2]
    coords = rand(batchSize, nCities, 2, 'single');
    % Permute to [C=2, B, T=N] for dlarray
    X = permute(coords,[3 1 2]);
    dlX = dlarray(X, 'CBT');

    % --- Actor Forward Pass (GNN) & Sampling ---
    [tours, logProbs, entropy] = actorStep_GNN(actorNet, dlX, numGNNLayers, nCities, nodeEmbSize, temperature);

    % --- Compute Rewards ---
    lengths = computeLengths(coords, tours);
    rewards = -lengths; % Use basic negative length for reward signal

    % --- Reward Normalization (Optional but often useful for Critic) ---
    useNormalizedReward = true;
    if useNormalizedReward
        rewards_mean = mean(rewards);
        rewards_std = std(rewards, 1);
        if rewards_std > 1e-8
            normalized_rewards = (rewards - rewards_mean) / rewards_std;
        else
            normalized_rewards = rewards - rewards_mean;
        end
        rewards_for_critic = normalized_rewards;
    else
        rewards_for_critic = rewards;
    end

    % --- Critic Baseline Estimate (using GNN output) ---
    nodeEmbeddings = gnnForward(actorNet, dlX, numGNNLayers, nCities, nodeEmbSize); % [Emb, B, N] 'CBT'
    % Pool embeddings over Time/Nodes dimension (dim 3) -> [Emb, B, 1]
    pooledEmbeddings = mean(nodeEmbeddings, 3);
    % Ensure correct label 'CBT' [C=Emb, B=Batch, T=1] for Critic input
    criticInputPooled = dlarray(extractdata(pooledEmbeddings), 'CBT');

    baselineDL = forward(criticNet, criticInputPooled); % Critic output [1, B, 1] 'CBT'
    baseline = squeeze(extractdata(baselineDL));

    % --- Advantage Calculation & Normalization ---
    adv = rewards - baseline(:);
    adv_mean = mean(adv);
    adv_std = std(adv, 1);
    if adv_std > 1e-8
        adv = (adv - adv_mean) / adv_std;
    else
        adv = adv - adv_mean;
    end

    % --- Compute Losses and Gradients ---
    [gradA, lossA] = dlfeval(@actorLoss_GNN, actorNet, dlX, adv, tours, entropyCoefficient, numGNNLayers, nCities, nodeEmbSize, temperature);
    % Pass POOLED embeddings (criticInputPooled) to criticLoss
    [gradC, lossC] = dlfeval(@criticLoss, criticNet, criticInputPooled, rewards_for_critic);

    % --- Gradient Clipping ---
    gradA = thresholdL2Norm( gradA, gradThreshold);
    gradC = thresholdL2Norm( gradC, gradThreshold);

    % --- Update Networks using Adam ---
    [actorNet, avgGradA, avgSqGradA] = adamupdate(...
        actorNet, gradA, avgGradA, avgSqGradA, epoch, learnRateA);
    [criticNet, avgGradC, avgSqGradC] = adamupdate(...
        criticNet, gradC, avgGradC, avgSqGradC, epoch, learnRateC);

    % --- Store and Display Progress & Early Stopping ---
    meanRewardEpoch = mean(rewards); % Use original rewards for reporting/stopping
    historyReward(epoch) =  meanRewardEpoch;
    if mod(epoch, 20) == 0 || epoch == 1
        elapsedTime = toc;
        fprintf('Epoch %d/%d, ActLoss=%.4f, CritLoss=%.4f, AvgReward=%.3f (Best: %.3f, Patience: %d/%d), Time=%.2fs\n', ...
            epoch, numEpochs, double(gather(lossA)), double(gather(lossC)), ...
            meanRewardEpoch, bestReward, epochsSinceImprovement, patience, elapsedTime);
        tic;
    end

    % Early Stopping Check (based on original reward scale)
    if meanRewardEpoch > bestReward + 1e-5
        bestReward = meanRewardEpoch;
        epochsSinceImprovement = 0;
        bestActorNetState = actorNet.Learnables;
        bestCriticNetState = criticNet.Learnables;
    else
        epochsSinceImprovement = epochsSinceImprovement + 1;
    end
    if epochsSinceImprovement >= patience
        fprintf('\nEarly stopping triggered after %d epochs with no improvement.\n', patience);
        fprintf('Best average reward achieved: %.4f\n', bestReward);
        if ~isempty(bestActorNetState)
            actorNet.Learnables = bestActorNetState;
            criticNet.Learnables = bestCriticNetState;
            fprintf('Restored model state from epoch with best reward.\n');
        else
             fprintf('Warning: No improvement detected, keeping last model state.\n');
        end
        historyReward = historyReward(1:epoch);
        break;
    end

end % End RL Training Loop

% --- Post-training Checks/Restore ---
if epoch == numEpochs && ~isempty(bestActorNetState) && epochsSinceImprovement > 0
    fprintf('\nRL Training finished. Restoring model state from epoch with best reward (%.4f).\n', bestReward);
    actorNet.Learnables = bestActorNetState;
    criticNet.Learnables = bestCriticNetState;
elseif epoch == numEpochs
     fprintf('\nRL Training finished.\n');
end

%% 5) Greedy Inference with Monte Carlo Sampling (GNN based)
fprintf('Running greedy inference with the trained GNN actor...\n');
coordsTest = rand(1, nCities, 2, 'single'); % Single instance B=1
dlXtest   = dlarray(permute(coordsTest,[3 1 2]), 'CBT'); % [C=2, B=1, T=N]

% Monte Carlo parameters
numMCSamples = 500 ; % Number of tours to sample
bestTourPred = [];
bestLenPred = Inf;

fprintf('Using inference temperature: %.2f, MC Samples: %d\n', inferenceTemperature, numMCSamples);

% Run multiple times with sampling (using inferenceTemperature) and keep the best result
for i = 1:numMCSamples
    % Use actorStep_GNN for sampling, but with inference temperature
    [tourPredSample, ~, ~] = actorStep_GNN(actorNet, dlXtest, numGNNLayers, nCities, nodeEmbSize, inferenceTemperature);
    lenPredSample = computeLengths(coordsTest, tourPredSample);

    if lenPredSample < bestLenPred
        bestLenPred = lenPredSample;
        bestTourPred = tourPredSample;
    end

    if mod(i, 50) == 0 % Update less frequently for MC
        fprintf('  MC samples: %d/%d, Current best: %.4f\n', i, numMCSamples, bestLenPred);
    end
end

% Use the best found tour from sampling
tourPred = bestTourPred;
lenPred = bestLenPred;

fprintf('RL GNN Best Sampled tour length: %.4f\n', lenPred);
fprintf('RL GNN Best Sampled tour sequence: %s\n', num2str(tourPred));

% --- Run Greedy Decode for Comparison (Optional) ---
% tourPredGreedy = greedyDecode_GNN(actorNet, dlXtest, numGNNLayers, nCities, nodeEmbSize);
% lenPredGreedy = computeLengths(coordsTest, tourPredGreedy);
% fprintf('RL GNN Greedy tour length: %.4f\n', lenPredGreedy);
% ---

%% --- Add Heuristic Comparison ---
fprintf('Running 2-opt heuristic...\n');
coordsTestMatrix = squeeze(double(coordsTest));
rlStartCity = tourPred(1); % Start 2-opt from where RL started
[tourHeuristic, lenHeuristic] = solveTSP_2opt(coordsTestMatrix, rlStartCity);
fprintf('2-opt Heuristic tour length: %.4f\n', lenHeuristic);
fprintf('2-opt Heuristic tour sequence: %s\n', num2str(tourHeuristic));
if lenHeuristic > 1e-9
    optimalityGap = ((lenPred - lenHeuristic) / lenHeuristic) * 100;
    fprintf('RL GNN (Sampled) solution gap vs 2-opt: %.2f%%\n', optimalityGap);
else
    fprintf('Heuristic length is near zero, cannot calculate gap.\n');
end

%% 6) Plotting (Training History and Tours)
% --- Plot Training History ---
figure;
actualEpochs = length(historyReward);
plot(1:actualEpochs, historyReward(1:actualEpochs), '-o', 'MarkerSize', 4);
if actualEpochs < numEpochs && ~isempty(bestActorNetState)
    [~, bestEpochIdx] = max(historyReward(1:actualEpochs));
    hold on;
    plot(bestEpochIdx, bestReward, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    text(bestEpochIdx, bestReward, sprintf(' Best: %.3f (Epoch %d)', bestReward, bestEpochIdx), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'r');
    hold off;
    title(sprintf('GNN RL Training Progress (Stopped Early at Epoch %d)', actualEpochs));
else
     title('GNN RL Training Progress');
end
xlabel('Epoch'); ylabel('Average Reward (Negative Length)'); grid on;
xlim([0, actualEpochs + 1]);

% --- Plot Tours ---
figure; hold on;
coordsPlot = squeeze(double(coordsTest));
tourPlotRL = double(squeeze(tourPred)); % Use best sampled tour
tourPlotHeuristic = double(squeeze(tourHeuristic));
% Plot Heuristic
orderedCoordsHeuristic = coordsPlot(tourPlotHeuristic, :);
pathCoordsHeuristic = [orderedCoordsHeuristic; orderedCoordsHeuristic(1,:)];
plot(pathCoordsHeuristic(:,1), pathCoordsHeuristic(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('2-opt (L=%.3f)', lenHeuristic));
% Plot RL
orderedCoordsRL = coordsPlot(tourPlotRL, :);
pathCoordsRL = [orderedCoordsRL; orderedCoordsRL(1,:)];
plot(pathCoordsRL(:,1), pathCoordsRL(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', sprintf('RL GNN Sampled (L=%.3f)', lenPred));
% Plot Cities & Labels
plot(coordsPlot(:,1), coordsPlot(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
for i = 1:nCities
    text(coordsPlot(i,1) + 0.01, coordsPlot(i,2) + 0.01, num2str(i), 'Color', 'k', 'FontSize', 10);
end
% Plot Start City
startCityIndex = tourPlotRL(1);
plot(coordsPlot(startCityIndex, 1), coordsPlot(startCityIndex, 2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Start (RL)');
hold off; title('RL GNN (Sampled) vs 2-opt Heuristic TSP Tours'); xlabel('X'); ylabel('Y');
legend('show', 'Location', 'best'); axis equal; grid on;
pad = 0.05; xlim([min(coordsPlot(:,1))-pad, max(coordsPlot(:,1))+pad]); ylim([min(coordsPlot(:,2))-pad, max(coordsPlot(:,2))+pad]);

% --- End of Main Script ---


%% --- Helper Functions ---

%% --- NEW Pre-training Loss Function ---
function [gradients, loss] = pretrainLoss(net, dlX, targetTours, numGNNLayers, nCities, nodeEmbSize)
    % dlX: [C, B, T], targetTours: [B, T] (uint32)
    B = size(dlX, 2);
    N = size(dlX, 3);

    % 1. Get GNN embeddings
    nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize); % [Emb, B, N]

    % --- Extract Weights ---
    Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
    Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

    % --- Calculate Keys ---
    keys = fullyconnect(nodeEmbeddings, Wk, Bk); % [Emb, B, N]

    % --- Initialize Decoder Variables ---
    mask = false(1, B, N);
    totalLoss = dlarray(0, 'like', dlX); % Initialize total loss for the batch
    meanEmbedding = mean(nodeEmbeddings, 3);
    currentContext = meanEmbedding;
    scaleFactor = sqrt(single(nodeEmbSize));

    % --- Loop through steps, calculating loss against target ---
    for t = 1:N
        % --- Generate Query & Scores (same as actorStep_GNN) ---
        query_input = reshape(currentContext, nodeEmbSize, B);
        query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB");
        reshaped_query = reshape(query, nodeEmbSize, B, 1);
        scores = sum(keys .* reshaped_query, 1); % -> [1, B, N]
        scores = scores / scaleFactor;

        % Optional: Layer Norm (keep consistent with RL phase if used there)
        % scores_mean = mean(scores, 3);
        % scores_std = std(scores, 0, 3) + 1e-5;
        % scores = (scores - scores_mean) ./ scores_std;

        % Apply mask
        scores(mask) = -inf;

        % --- Calculate Log Softmax Probabilities ---
        % Need log probabilities for cross-entropy loss
        % Use logsoftmax for numerical stability
        logProbs_step = logsoftmax(scores, 3); % LogSoftmax over Cities dim -> [1, B, N]

        % --- Calculate Loss for this Step ---
        % Target index for this step
        targetIdx = targetTours(:, t); % [B, 1]

        % Indices into logProbs [1, B, N] corresponding to targets
        linearIndicesTargets = sub2ind(size(logProbs_step), ones(B, 1), (1:B)', double(targetIdx));

        % Negative log-likelihood loss (Cross-Entropy) for this step
        stepLoss = -logProbs_step(linearIndicesTargets); % Extract logProb of target -> [B, 1]
        totalLoss = totalLoss + mean(stepLoss); % Accumulate mean loss over batch

        % --- Update Mask and Context using TARGET tour ---
        linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(targetIdx));
        mask(linearIndicesMask) = true;

        linearIndicesEmb = sub2ind([B, N], (1:B)', double(targetIdx));
        featuresSelected = nodeEmbeddings(:, linearIndicesEmb);
        currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT');

    end % End step loop

    loss = totalLoss / N; % Average loss over all steps
    gradients = dlgradient(loss, net.Learnables);
end


%% --- GNN Forward Pass Helper ---
function nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, nCities, nodeEmbSize)
    % (Code from previous GNN implementation - unchanged)
    B = size(dlX, 2);
    persistent adjHat_static
    persistent nCities_static
    if isempty(adjHat_static) || nCities ~= nCities_static
        adj = ones(nCities, nCities, 'single') - eye(nCities, 'single');
        adjWithSelfLoops = adj + eye(nCities, 'single');
        degree = sum(adjWithSelfLoops, 2);
        degreeInvSqrt = diag(1./sqrt(degree));
        degreeInvSqrt(isinf(degreeInvSqrt)) = 0;
        adjHat_static = degreeInvSqrt * adjWithSelfLoops * degreeInvSqrt;
        adjHat_static = dlarray(adjHat_static);
        nCities_static = nCities;
    end
    adjHat = adjHat_static;

    embW = net.Learnables{strcmp(net.Learnables.Layer,'nodeEmbFC') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    embB = net.Learnables{strcmp(net.Learnables.Layer,'nodeEmbFC') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
    dlX_perm = permute(stripdims(dlX), [1 3 2]);
    nodeH = fullyconnect(dlX_perm, embW, embB, 'DataFormat', 'CBT');
    nodeH = relu(nodeH);
    nodeH = permute(nodeH, [1 3 2]);
    % nodeH = dlarray(nodeH, 'CBT');

    for i = 1:numGNNLayers
        layerName = ['gnnFC' num2str(i)];
        gnnW = net.Learnables{strcmp(net.Learnables.Layer, layerName) & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
        gnnB = net.Learnables{strcmp(net.Learnables.Layer, layerName) & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
        nodeH_perm_BST = permute(stripdims(nodeH), [2 3 1]);
        nodeHW = pagemtimes(nodeH_perm_BST, gnnW');
        nodeHW_perm_NEB = permute(nodeHW, [2 3 1]);
        adjHat_nodeHW = pagemtimes(adjHat, nodeHW_perm_NEB);
        bias_reshaped = reshape(gnnB, 1, nodeEmbSize, 1);
        adjHat_nodeHW_bias = adjHat_nodeHW + bias_reshaped;
        out_perm = relu(adjHat_nodeHW_bias);
        nodeH = permute(out_perm, [2 3 1]);
        
    end
    nodeH = dlarray(nodeH, 'CBT');
    nodeEmbeddings = nodeH;
end


%% --- GNN Actor Step ---
function [tours, logProbs, entropy] = actorStep_GNN(net, dlX, numGNNLayers, nCities, nodeEmbSize, temperature)
    % (Code from previous GNN implementation - added temperature argument)
    B = size(dlX, 2);
    N = size(dlX, 3);

    nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize);

    Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
    Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

    keys = fullyconnect(nodeEmbeddings, Wk, Bk);

    mask = false(1, B, N);
    tours = zeros(B, N, 'uint32');
    logProbs = zeros(B, N, 'like', dlX);
    entropy_steps = zeros(1, B, N, 'like', dlX);
    meanEmbedding = mean(nodeEmbeddings, 3);
    currentContext = meanEmbedding;
    scaleFactor = sqrt(single(nodeEmbSize));

    for t = 1:N
        query_input = reshape(currentContext, nodeEmbSize, B);
        query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB");
        reshaped_query = reshape(query, nodeEmbSize, B, 1);
        scores = sum(keys .* reshaped_query, 1);
        scores = scores / scaleFactor;

        % Optional Layer Norm
        scores_mean = mean(scores, 3);
        scores_std = std(scores, 0, 3) + 1e-5;
        scores = (scores - scores_mean) ./ scores_std;

        scores(mask) = -inf;

        scores_ = stripdims(scores);
        scores_scaled = scores_ / temperature; % Apply temperature
        probs = softmax(scores_scaled, "DataFormat", "TBC");

        idx = zeros(B, 1, 'uint32');
        probs_gathered = gather(extractdata(probs));
        scores_gathered = gather(extractdata(scores));
        mask_gathered = gather(extractdata(mask)); % Need to gather mask too

        for i = 1:B
            current_probs_raw = squeeze(probs_gathered(1, i, :));
            p = max(0, current_probs_raw);
             p_sum = sum(p);
            if p_sum > 1e-8
                 p = p / p_sum;
            else
                 p(:) = 1/N;
            end

            if any(isnan(p)) || sum(p) < 1e-6
                current_mask = squeeze(mask_gathered(1, i, :)); % Use gathered mask
                valid_indices = find(~current_mask);
                if isempty(valid_indices)
                    current_scores = squeeze(scores_gathered(1, i, :));
                    [~, max_idx_val] = max(current_scores);
                     idx(i) = max_idx_val;
                else
                    current_scores_valid = squeeze(scores_gathered(1, i, valid_indices));
                    [~, max_idx_local] = max(current_scores_valid);
                    idx(i) = valid_indices(max_idx_local);
                end
            else
                 idx(i) = randsample(N, 1, true, p(:));
            end
        end

        tours(:,t) = idx;
        linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
        logProbs(:,t) = log(probs(linearIndicesProbs) + 1e-10);

        stepEntropy = -sum(probs .* log(probs + 1e-10), 3);
        entropy_steps(1, :, t) = stepEntropy;

        linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(idx));
        mask(linearIndicesMask) = true; % Update dlarray mask

        linearIndicesEmb = sub2ind([B, N], (1:B)', double(idx));
        featuresSelected = nodeEmbeddings(:, linearIndicesEmb);
        currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT');

    end
    entropy = reshape(sum(entropy_steps, 3), B, 1);
end


%% --- GNN Greedy Decode ---
function tours = greedyDecode_GNN(net, dlX, numGNNLayers, nCities, nodeEmbSize, temperature)
    % (Code from previous GNN implementation - Added temperature argument, sampling logic)
     if nargin < 6, temperature = 1.0; end % Default temperature

     B = size(dlX, 2); % Should be 1
     N = size(dlX, 3);

     nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize);

     Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
     Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
     Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
     Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

     keys = fullyconnect(nodeEmbeddings, Wk, Bk);

     mask = false(1, B, N);
     tours = zeros(B, N, 'uint32');
     meanEmbedding = mean(nodeEmbeddings, 3);
     currentContext = meanEmbedding;
     scaleFactor = sqrt(single(nodeEmbSize));

     for t = 1:N
         query_input = reshape(currentContext, nodeEmbSize, B);
         query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB");
         reshaped_query = reshape(query, nodeEmbSize, B, 1);
         scores = sum(keys .* reshaped_query, 1);
         scores = scores / scaleFactor;

         % Layer normalization
         scores_mean = mean(scores, 3);
         scores_std = std(scores, 0, 3) + 1e-5;
         scores = (scores - scores_mean) ./ scores_std;

         scores(mask) = -inf;

         % --- Select Next City (Greedy or Temp Sampling) ---
         if temperature < 0.99 && t < N % Use temperature sampling for non-last steps if temp < 1
             scores_ = stripdims(scores);
             scores_scaled = scores_ / temperature;
             probs = softmax(scores_scaled, "DataFormat", "TBC");
             probs_gathered = gather(extractdata(probs));
             p = squeeze(probs_gathered(1, 1, :));
             p = max(0, p);
             p_sum = sum(p);
             if p_sum > 1e-8
                  p = p / p_sum;
             else % Fallback if all probabilities are zero
                  p(:) = 1/N; % Uniform distribution
             end
              if any(isnan(p)) % Additional fallback
                  p(:) = 1/N;
              end
             idx = randsample(N, 1, true, p(:)); % Ensure p is vector
             idx = uint32(idx);
         else % Use standard greedy (argmax)
             [~, idx] = max(scores, [], 3);
             idx = squeeze(idx);
             idx = uint32(extractdata(idx));
         end

         tours(:,t) = idx;
         linearIndicesMask = sub2ind(size(mask), 1, B, double(idx)); % B is 1 here
         mask(linearIndicesMask) = true;

         linearIndicesEmb = sub2ind([B, N], 1, double(idx)); % B is 1 here
         featuresSelected = nodeEmbeddings(:, linearIndicesEmb);
         currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT');
     end
end


%% --- GNN Actor Loss ---
function [gradients, loss] = actorLoss_GNN(net, dlX, adv, tours, entropyCoeff, numGNNLayers, nCities, nodeEmbSize, temperature)
    % Re-calculate logProbs and entropy using GNN model inside dlfeval
    [logProbs, entropy] = calculateLogProbsAndEntropyForTours_GNN(net, dlX, tours, numGNNLayers, nCities, nodeEmbSize, temperature);

    sumLogProbs = sum(logProbs, 2);
    adv = dlarray(adv(:));
    policyLoss = -mean(sumLogProbs .* adv);
    entropyLoss = -mean(entropy);
    loss = policyLoss + entropyCoeff * entropyLoss;
    gradients = dlgradient(loss, net.Learnables);
end


%% --- GNN LogProb/Entropy Calculation Helper ---
function [logProbs, entropy] = calculateLogProbsAndEntropyForTours_GNN(net, dlX, tours, numGNNLayers, nCities, nodeEmbSize, temperature)
    % Mirrors actorStep_GNN but uses provided tours, added temperature
    if nargin < 7, temperature = 1.0; end % Default temperature if not passed

    B = size(dlX, 2);
    N = size(dlX, 3);

    nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize);

    Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
    Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

    keys = fullyconnect(nodeEmbeddings, Wk, Bk);

    mask = false(1, B, N);
    logProbs = zeros(B, N, 'like', dlX);
    entropy_steps = zeros(1, B, N, 'like', dlX);
    meanEmbedding = mean(nodeEmbeddings, 3);
    currentContext = meanEmbedding;
    scaleFactor = sqrt(single(nodeEmbSize));

    for t = 1:N
        query_input = reshape(currentContext, nodeEmbSize, B);
        query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB");
        reshaped_query = reshape(query, nodeEmbSize, B, 1);
        scores = sum(keys .* reshaped_query, 1);
        scores = scores / scaleFactor;

        % Layer norm
        scores_mean = mean(scores, 3);
        scores_std = std(scores, 0, 3) + 1e-5;
        scores = (scores - scores_mean) ./ scores_std;

        scores(mask) = -inf;

        scores_ = stripdims(scores);
        scores_scaled = scores_ / temperature; % Apply temperature
        probs = softmax(scores_scaled, "DataFormat", "TBC");

        idx = tours(:, t);

        linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
        logProbs(:,t) = log(probs(linearIndicesProbs) + 1e-10);

        stepEntropy = -sum(probs .* log(probs + 1e-10), 3);
        entropy_steps(1, :, t) = stepEntropy;

        linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(idx));
        mask(linearIndicesMask) = true;

        linearIndicesEmb = sub2ind([B, N], (1:B)', double(idx));
        featuresSelected = nodeEmbeddings(:, linearIndicesEmb);
        currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT');

    end
    entropy = reshape(sum(entropy_steps, 3), B, 1);
end


%% --- Other Helper Functions ---

% Critic Loss
function [gradients, loss] = criticLoss(net, criticInputPooled, rewards)
    % criticInputPooled should be [Emb, 1, Batch] 'CBT' from GNN pooling
    baselinePredicted = forward(net, criticInputPooled); % Output [1, 1, B] 'CBT'
    baselinePredicted = squeeze(baselinePredicted);
    rewards = dlarray(rewards(:));
    baselinePredicted = baselinePredicted(:);
    loss = mean((baselinePredicted - rewards).^2);
    gradients = dlgradient(loss, net.Learnables);
end

% Threshold L2 Norm
function clippedGradients = thresholdL2Norm(gradientsTable, threshold)
    gradNorm = single(0);
    numGrads = height(gradientsTable);
    gradientValues = cell(numGrads, 1);
    for i = 1:numGrads
        currentGrad = gradientsTable.Value{i};
        if ~isempty(currentGrad) && isa(currentGrad, 'dlarray')
            gradientValues{i} = currentGrad;
            gradNorm = gradNorm + sum(currentGrad(:).^2);
        end
    end
    gradNorm = sqrt(gradNorm);
    clippedGradients = gradientsTable;
    if gradNorm > threshold
        normScale = threshold / gradNorm;
        for i = 1:numGrads
             if ~isempty(gradientValues{i})
                clippedGradients.Value{i} = gradientValues{i} * normScale;
             end
        end
    end
end

% Compute Lengths
function L = computeLengths(coords, tours)
    B = size(coords,1);
    N = size(coords,2);
    if isa(coords, 'dlarray')
        coords = extractdata(coords); end
    if ~isa(coords, 'single') && ~isa(coords, 'double')
         coords = double(coords); end
    if ~isa(tours, 'double')
        tours = double(tours); end
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

% --- 2-Opt Heuristic Functions ---
function [bestTour, bestLength] = solveTSP_2opt(coords, startCity)
    if nargin < 2, startCity = 1; end
    n = size(coords, 1);
    if n <= 3
        bestTour = [startCity, setdiff(1:n, startCity)];
        bestLength = calculateTourLength(coords, bestTour);
        return;
    end
    currentTour = nearestNeighborTour(coords, startCity);
    currentLength = calculateTourLength(coords, currentTour);
    bestTour = currentTour;
    bestLength = currentLength;
    improved = true;
    while improved
        improved = false;
        for i = 1 : n-1
            for k = i+1 : n
                newTour = twoOptSwap(currentTour, i, k);
                newLength = calculateTourLength(coords, newTour);
                if newLength < currentLength - 1e-6
                    currentTour = newTour;
                    currentLength = newLength;
                    improved = true;
                    if currentLength < bestLength
                        bestLength = currentLength;
                        bestTour = currentTour;
                    end
                end
            end
        end
    end
end

function len = calculateTourLength(coords, tour)
    orderedCoords = coords(tour,:);
    pathCoords = [orderedCoords; orderedCoords(1,:)];
    diffs = diff(pathCoords, 1, 1);
    segmentLengths = sqrt(sum(diffs.^2, 2));
    len = sum(segmentLengths);
end

function newTour = twoOptSwap(tour, i, k)
    newTour = tour;
    segmentToReverse = newTour(i+1 : k);
    newTour(i+1 : k) = fliplr(segmentToReverse);
end

function tour = nearestNeighborTour(coords, startCity)
    if nargin < 2, startCity = 1; end
    n = size(coords, 1);
    tour = zeros(1, n);
    visited = false(1, n);
    currentCity = startCity;
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