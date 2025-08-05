%% Neural Combinatorial Optimization (NCO) example, using GNN model
% # Traveling Salesman Problem (TSP) Solver Using Reinforcement Learning
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
%    - Define critic network for value estimation (using GNN output)
%    - Initialize both networks
%
% 3. **Training Loop**
%    - Generate random TSP instances
%    - Run GNN-based actor for tour sampling (using attention over node embeddings)
%    - Compute tour lengths and corresponding rewards (with potential curriculum/normalization)
%    - Use critic to estimate baseline values from GNN embeddings
%    - Calculate advantage (reward - baseline)
%    - Compute actor and critic losses
%    - Apply gradient clipping and parameter updates
%    - Track performance and apply early stopping
%
% 4. **Inference and Evaluation**
%    - Perform greedy decoding using GNN-based actor
%    - Compare with 2-opt heuristic solution
%    - Calculate optimality gap
%    - Visualize tours for comparison
%
% 5. **Helper Functions**
%    - GNN Forward Pass (`gnnForward`)
%    - Tour length calculation (`computeLengths`)
%    - Gradient clipping (`thresholdL2Norm`)
%    - Actor and critic loss functions (`actorLoss_GNN`, `criticLoss`)
%    - GNN-based step functions (`actorStep_GNN`, `greedyDecode_GNN`, `calculateLogProbsAndEntropyForTours_GNN`)
%    - 2-opt local search implementation (`solveTSP_2opt`, helpers)
%

% Copyright 2025, The MathWorks, Inc.

%% Hyperparameters
nCities    = 20;       % Number of cities
nodeEmbSize = 32;      % GNN node embedding dimension (equivalent to hiddenSize)
numGNNLayers = 1;      % Number of GNN layers to stack
batchSize  = 8;        % Minibatch size
numEpochs  = 1000;     % Training epochs
gradThreshold = 2.0;   % L2 norm threshold for gradient clipping

% Pre-training Parameters
numPreTrainEpochs = 100; % Number of epochs for pre-training
preTrainBatchSize = 64; % Larger batch size for pre-training
preTrainLR = 1e-3;     % Learning rate for pre-training

% Learning Rates
learnRateA = 1e-2; % Actor LR (adjusted slightly from original, might need tuning)
learnRateC = 2e-2; % Critic LR (adjusted slightly from original)

% Entropy Coefficient
entropyCoefficient = 0.1; % Coefficient for entropy bonus in actor loss

% Temperature Parameter
temperature = 1.2;      % Controls softmax sharpness (1.0=standard, <1.0=sharper, >1.0=smoother)

% Early Stopping Parameters
patiencePercentage = 0.30; % Stop if no improvement for 20% of total epochs
patience = ceil(numEpochs * patiencePercentage);

% Optional: For reproducibility
rng(1);

%% 1) Define Pointer (Actor) Network (GNN based)
actorLG = layerGraph(); % Use a temporary variable for the layer graph

% Input Layer (Coordinates)
inputLayer = sequenceInputLayer(2, 'Name', 'in', 'Normalization', 'none');
actorLG = addLayers(actorLG, inputLayer);

% Initial Node Embedding Layer
nodeEmbFCLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'nodeEmbFC');
nodeEmbReluLayer = reluLayer('Name', 'nodeEmbRelu');
actorLG = addLayers(actorLG, nodeEmbFCLayer);
actorLG = addLayers(actorLG, nodeEmbReluLayer);

% Connect input to embedding
actorLG = connectLayers(actorLG, 'in', 'nodeEmbFC');
actorLG = connectLayers(actorLG, 'nodeEmbFC', 'nodeEmbRelu');

% Keep track of the last layer name for sequential connection
lastGNNInputLayerName = 'nodeEmbRelu'; % Input to the first GNN layer
lastGNNOutputLayerName = ''; % Will store the name of the final GNN layer

% Define AND Connect GNN Layers sequentially
gnnLayerNames = strings(1, numGNNLayers);
for i = 1:numGNNLayers
    gnnLayerName = ['gnnFC' num2str(i)];
    gnnLayerNames(i) = gnnLayerName;
    gnnFCLayer = fullyConnectedLayer(nodeEmbSize, 'Name', gnnLayerName);
    actorLG = addLayers(actorLG, gnnFCLayer);
    % Connect from the previous layer in the chain
    actorLG = connectLayers(actorLG, lastGNNInputLayerName, gnnLayerName);
    lastGNNInputLayerName = gnnLayerName; % Update for next connection
    lastGNNOutputLayerName = gnnLayerName; % Track the last one added
end

% Define FC Layers for Attention Projections (K, V, Q)
fcKeyLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'fcKey');
fcValueLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'fcValue');
fcQueryLayer = fullyConnectedLayer(nodeEmbSize, 'Name', 'fcQuery');
actorLG = addLayers(actorLG, fcKeyLayer);
actorLG = addLayers(actorLG, fcValueLayer);
actorLG = addLayers(actorLG, fcQueryLayer);

% --- Connect Attention Layers ---
% Connect them to the output of the last GNN layer to satisfy graph requirements.
% Even if we don't use forward(net, ..., 'Outputs', 'fcKey'), this makes the graph valid.
if ~isempty(lastGNNOutputLayerName) % Check if GNN layers were actually added
    actorLG = connectLayers(actorLG, lastGNNOutputLayerName, 'fcKey');
    actorLG = connectLayers(actorLG, lastGNNOutputLayerName, 'fcValue');
    actorLG = connectLayers(actorLG, lastGNNOutputLayerName, 'fcQuery');
else
    % If no GNN layers (numGNNLayers=0), connect from embedding relu
    actorLG = connectLayers(actorLG, 'nodeEmbRelu', 'fcKey');
    actorLG = connectLayers(actorLG, 'nodeEmbRelu', 'fcValue');
    actorLG = connectLayers(actorLG, 'nodeEmbRelu', 'fcQuery');
end
% --------------------------------

% Create the dlnetwork from the connected graph
actorNet = dlnetwork(actorLG);

% Initialize Actor
dummyX_GNN = dlarray(rand(2, batchSize, nCities, 'single'), 'CBT');
actorNet = initialize(actorNet, dummyX_GNN);
fprintf('Actor network initialized (GNN based).\n');

%% 2) Define Critic Network (Takes GNN Output Sequence)
criticLayers = layerGraph(); % Start empty

% Input: GNN Node Embeddings [EmbSize, Batch, NumNodes] ('CBT')
criticLayers = addLayers(criticLayers, sequenceInputLayer(nodeEmbSize, 'Name', 'criticGNNIn', 'Normalization', 'none'));

% Global Average Pooling over Nodes ('T' dimension)
criticLayers = addLayers(criticLayers, globalAveragePooling1dLayer('Name', 'criticPool'));
criticLayers = connectLayers(criticLayers, 'criticGNNIn', 'criticPool');

% MLP Part (using nodeEmbSize)
criticLayers = addLayers(criticLayers, fullyConnectedLayer(nodeEmbSize * 2, 'Name', 'cFC1')); % Match original spec
criticLayers = connectLayers(criticLayers, 'criticPool', 'cFC1');

criticLayers = addLayers(criticLayers, reluLayer('Name', 'cRelu1'));
criticLayers = connectLayers(criticLayers, 'cFC1', 'cRelu1');

criticLayers = addLayers(criticLayers, fullyConnectedLayer(nodeEmbSize, 'Name', 'cFC2')); % Match original spec
criticLayers = connectLayers(criticLayers, 'cRelu1', 'cFC2');

criticLayers = addLayers(criticLayers, reluLayer('Name', 'cRelu2'));
criticLayers = connectLayers(criticLayers, 'cFC2', 'cRelu2');

criticLayers = addLayers(criticLayers, fullyConnectedLayer(1, 'Name', 'cOut'));
criticLayers = connectLayers(criticLayers, 'cRelu2', 'cOut');

criticNet = dlnetwork(criticLayers);

% Initialize Critic - Input is now sequence [Emb, Batch, Nodes]
dummyCriticInGNN = dlarray(rand(nodeEmbSize, batchSize, nCities, 'single'), 'CBT');
criticNet = initialize(criticNet, dummyCriticInGNN);
fprintf('Critic network initialized (takes GNN output).\n');

%% 2.5) Pre-train Actor with 2-opt Solutions
fprintf('Pre-training actor network with 2-opt solutions...\n');
[actorNet, avgGradA, avgSqGradA] = preTrainWithTwoOpt(actorNet, numPreTrainEpochs, preTrainBatchSize, nCities, nodeEmbSize, numGNNLayers, preTrainLR, temperature);

%% 3) Training Loop (with modified initialization)
% The avgGradA and avgSqGradA are now initialized from pre-training
% avgGradC and avgSqGradC still need initialization:
avgGradC = []; avgSqGradC = [];

% --- Early Stopping Parameters ---
bestReward = -inf;
epochsSinceImprovement = 0;
bestActorNetState = [];
bestCriticNetState = [];
% --------------------------------

fprintf('Starting RL training (GNN Based, pre-trained with 2-opt, nCities=%d)...\n', nCities);
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
    [tours, ~, entropy] = actorStep_GNN(actorNet, dlX, numGNNLayers, nCities, nodeEmbSize, temperature);

    % --- Compute Rewards ---
    lengths = computeLengths(coords, tours);

    % Curriculum learning / Reward Shaping (optional)
    if epoch <= numEpochs * 0.3
        rewards = -lengths;  % Basic negative length
    else
        % Add bonus for better tours compared to batch average
        avgLength = mean(lengths);
        bonusFactor = 0.2; % Tune this factor
        rewards = -lengths + bonusFactor * max(0, (avgLength - lengths));
    end

    % --- Reward Normalization (Optional) ---
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
        rewards_for_critic = rewards; %#ok<*UNRCH>
    end

    % --- Critic Baseline Estimate (using GNN output) ---
    % Get final node embeddings from GNN [Emb, B, N] 'CBT'
    % Note: Re-run GNN. Can optimize by returning embeddings from actorStep_GNN
    nodeEmbeddings = gnnForward(actorNet, dlX, numGNNLayers, nCities, nodeEmbSize);
    criticInput = nodeEmbeddings; % Input for the sequence-input Critic

    baselineDL = forward(criticNet, criticInput); % Critic output [1, B, 1] 'CBT'
    baseline = squeeze(extractdata(baselineDL));

    % --- Advantage Calculation & Normalization ---
    % Use original (potentially shaped) rewards for advantage
    adv = rewards - baseline(:);
    adv_mean = mean(adv);
    adv_std = std(adv, 1);
    if adv_std > 1e-8
        adv = (adv - adv_mean) /  (adv_std + 1e-8); % Add epsilon
    else
        adv = zeros(size(adv)); % If std is too small, use zeros instead
    end

    % --- Compute Losses and Gradients ---
    [gradA, lossA] = dlfeval(@actorLoss_GNN, actorNet, dlX, adv, tours, entropyCoefficient, numGNNLayers, nCities, nodeEmbSize, temperature);
    % Pass GNN embeddings (criticInput) to criticLoss
    [gradC, lossC] = dlfeval(@criticLoss, criticNet, criticInput, rewards_for_critic);


    gradA = checkAndFixGradients(gradA);
    gradC = checkAndFixGradients(gradC);

    % --- Gradient Clipping ---
    gradA = thresholdL2Norm( gradA, gradThreshold);
    gradC = thresholdL2Norm( gradC, gradThreshold);

    % --- Update Networks using Adam ---
    [actorNet, avgGradA, avgSqGradA] = adamupdate(...
        actorNet, gradA, avgGradA, avgSqGradA, epoch, learnRateA); % Use learnRateA
    [criticNet, avgGradC, avgSqGradC] = adamupdate(...
        criticNet, gradC, avgGradC, avgSqGradC, epoch, learnRateC); % Use learnRateC

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

end % End Training Loop

% --- Post-training Checks/Restore ---
if epoch == numEpochs && ~isempty(bestActorNetState) && epochsSinceImprovement > 0
    fprintf('\nTraining finished. Restoring model state from epoch with best reward (%.4f).\n', bestReward);
    actorNet.Learnables = bestActorNetState;
    criticNet.Learnables = bestCriticNetState;
elseif epoch == numEpochs
    fprintf('\nTraining finished.\n');
end

%% 4) Greedy Inference Section - Modified to compare with pre-training baseline
fprintf('Evaluating different solution methods...\n');
coordsTest = rand(1, nCities, 2, 'single');
% Permute to [C=2, B=1, T=N]
dlXtest = dlarray(permute(coordsTest,[3 1 2]), 'CBT');

% Monte Carlo parameters
numSamples = 500;
bestTourPred = [];
bestLenPred = Inf;

% Run multiple times and keep the best result
for i = 1:numSamples
    tourPred = greedyDecode_GNN(actorNet, dlXtest, numGNNLayers, nCities, nodeEmbSize, temperature);
    lenPred = computeLengths(coordsTest, tourPred);

    if lenPred < bestLenPred
        bestLenPred = lenPred;
        bestTourPred = tourPred;
    end

    % Optional: Show progress every 100 samples
    if mod(i, 100) == 0
        fprintf('  Monte Carlo samples: %d/%d, Current best: %.4f\n', i, numSamples, bestLenPred);
    end
end

% Use the best found tour
tourPred = bestTourPred;
lenPred = bestLenPred;

fprintf('RL GNN Monte Carlo best tour length: %.4f (from %d samples)\n', lenPred, numSamples);

% --- Run 2-opt Comparison ---
fprintf('Running 2-opt heuristic...\n');
coordsTestMatrix = squeeze(double(coordsTest));
rlStartCity = tourPred(1); % Start 2-opt from where RL started for fairer comparison
[tourHeuristic, lenHeuristic] = solveTSP_2opt(coordsTestMatrix, rlStartCity);
fprintf('2-opt Heuristic tour length: %.4f\n', lenHeuristic);

% Calculate optimality gap
if lenHeuristic > 1e-9
    optimalityGap = ((lenPred - lenHeuristic) / lenHeuristic) * 100;
    fprintf('RL GNN solution gap vs 2-opt: %.2f%%\n', optimalityGap);
else
    fprintf('Heuristic length is near zero, cannot calculate gap.\n');
end

%% 5) Plotting (Training History and Tours)
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
    title(sprintf('GNN Training Progress (Stopped Early at Epoch %d)', actualEpochs));
else
    title('GNN Training Progress');
end
xlabel('Epoch'); ylabel('Average Reward (Negative Length)'); grid on;
xlim([0, actualEpochs + 1]);

% --- Plot Tours ---
figure; hold on;
coordsPlot = squeeze(double(coordsTest));
tourPlotRL = double(squeeze(tourPred));
tourPlotHeuristic = double(squeeze(tourHeuristic));
% Plot Heuristic
orderedCoordsHeuristic = coordsPlot(tourPlotHeuristic, :);
pathCoordsHeuristic = [orderedCoordsHeuristic; orderedCoordsHeuristic(1,:)];
plot(pathCoordsHeuristic(:,1), pathCoordsHeuristic(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('2-opt (L=%.3f)', lenHeuristic));
% Plot RL
orderedCoordsRL = coordsPlot(tourPlotRL, :);
pathCoordsRL = [orderedCoordsRL; orderedCoordsRL(1,:)];
plot(pathCoordsRL(:,1), pathCoordsRL(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', sprintf('RL GNN (L=%.3f)', lenPred));
% Plot Cities & Labels
plot(coordsPlot(:,1), coordsPlot(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
for i = 1:nCities
    text(coordsPlot(i,1) + 0.01, coordsPlot(i,2) + 0.01, num2str(i), 'Color', 'k', 'FontSize', 10);
end
% Plot Start City
startCityIndex = tourPlotRL(1);
plot(coordsPlot(startCityIndex, 1), coordsPlot(startCityIndex, 2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Start (RL)');
hold off; title('RL GNN vs 2-opt Heuristic TSP Tours'); xlabel('X'); ylabel('Y');
legend('show', 'Location', 'best'); axis equal; grid on;
pad = 0.05; xlim([min(coordsPlot(:,1))-pad, max(coordsPlot(:,1))+pad]); ylim([min(coordsPlot(:,2))-pad, max(coordsPlot(:,2))+pad]);

% --- End of Main Script ---


%% --- GNN Forward Pass Helper ---
function nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, nCities, nodeEmbSize)
% CHANGE: Add stability parameter
epsilon = 1e-10;

% Calculate Normalized Adjacency Matrix (cached)
persistent adjHat_static nCities_static
if isempty(adjHat_static) || nCities ~= nCities_static
    adj = ones(nCities, nCities, 'single') - eye(nCities, 'single');
    adjWithSelfLoops = adj + eye(nCities, 'single');
    degree = sum(adjWithSelfLoops, 2);

    % CHANGE: More stable calculation with epsilon
    degreeInvSqrt = diag(1./sqrt(degree + epsilon));
    degreeInvSqrt(isinf(degreeInvSqrt)) = 0;
    adjHat_static = degreeInvSqrt * adjWithSelfLoops * degreeInvSqrt;
    adjHat_static = dlarray(adjHat_static);
    nCities_static = nCities;
end
adjHat = adjHat_static;

% Initial Node Embedding
embW = net.Learnables{strcmp(net.Learnables.Layer,'nodeEmbFC') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
embB = net.Learnables{strcmp(net.Learnables.Layer,'nodeEmbFC') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
nodeH = fullyconnect(dlX, embW, embB);
nodeH = relu(nodeH);

noteH_dims = dims(nodeH);
[~, B, N] = size(nodeH);

% CHANGE: Add layer normalization before GNN passes
nodeH = layerNormalize(nodeH, 1, epsilon);

% GNN Layers
for i = 1:numGNNLayers
    layerName = ['gnnFC' num2str(i)];
    gnnW = net.Learnables{strcmp(net.Learnables.Layer, layerName) & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
    gnnB = net.Learnables{strcmp(net.Learnables.Layer, layerName) & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

    % CHANGE: More stable reshape with checks
    if any(isnan(nodeH),'all') || any(isinf(nodeH),'all')
        fprintf('Warning: NaN/Inf in nodeH before GNN layer %d\n', i);
        nodeH(isnan(nodeH) | isinf(nodeH)) = 0;
    end

    % 1. Calculate HW = H * W with more stable operations
    nodeH_reshaped = reshape(permute(stripdims(nodeH), [2 3 1]), [], nodeEmbSize);
    nodeHW_reshaped = nodeH_reshaped * gnnW';
    nodeHW = reshape(nodeHW_reshaped, B, N, nodeEmbSize);

    % 2. Calculate A_hat * (HW) with numerical stability
    nodeHW_perm_NEB = permute(nodeHW, [2 3 1]);
    adjHat_nodeHW = pagemtimes(adjHat, nodeHW_perm_NEB);

    % 3. Add Bias with better reshaping
    bias_reshaped = reshape(gnnB, 1, nodeEmbSize, 1);

    % CHANGE: Validate output before adding bias
    if any(isnan(adjHat_nodeHW(:))) || any(isinf(adjHat_nodeHW(:)))
        fprintf('Warning: NaN/Inf detected in GNN computation layer %d\n', i);
        adjHat_nodeHW(isnan(adjHat_nodeHW) | isinf(adjHat_nodeHW)) = 0;
    end

    adjHat_nodeHW_bias = adjHat_nodeHW + bias_reshaped;

    % CHANGE: Add layer normalization for better stability
    adjHat_nodeHW_bias = layerNormalize(adjHat_nodeHW_bias, 2, epsilon);

    % 4. Apply activation
    out_perm = relu(adjHat_nodeHW_bias);

    % 5. Permute back to standard format
    nodeH = permute(out_perm, [2 3 1]);
end

nodeEmbeddings = dlarray(nodeH, noteH_dims);
end

% CHANGE: Add helper for layer normalization
function x_norm = layerNormalize(x, dim, epsilon)
% Simple layer normalization along specified dimension
mu = mean(x, dim);
sigma = std(x, 0, dim);
x_norm = (x - mu) ./ (sigma + epsilon);
end

%% --- NEW GNN Actor Step ---
function [tours, logProbs, entropy] = actorStep_GNN(net, dlX, numGNNLayers, nCities, nodeEmbSize, temperature) %#ok<INUSD>
% dlX: [C=2, B, T=N]
B = size(dlX, 2);
N = size(dlX, 3); % Num Cities

% 1. --- Encoder: GNN Forward Pass ---
nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize); % [Emb, B, N] 'CBT'

% --- Extract Attention Projection Weights ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys (from GNN embeddings) ---
keys = fullyconnect(nodeEmbeddings, Wk, Bk); % Assumes 'CBT' -> [Emb, B, N]

% --- Initialize Decoder Loop Variables ---
mask = false(1, B, N); % [1, B, N]
tours = zeros(B, N, 'uint32'); % [B, N]
logProbs = zeros(B, N, 'like', dlX); % [B, N]
entropy_steps = zeros(1, B, N, 'like', dlX); % [1, B, N]

% Initial context: mean embedding [Emb, B, 1] 'CBT'
meanEmbedding = mean(nodeEmbeddings, 3);
currentContext = meanEmbedding;

scaleFactor = sqrt(single(nodeEmbSize));

% 2. --- "Decoder" Loop (City Selection) ---
for t = 1:N
    % --- Generate Query from Current Context ---
    query_input = reshape(currentContext, nodeEmbSize, B); % [Emb, B]
    query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB"); % [Emb, B]

    % --- Attention Scores ---
    reshaped_query = reshape(query, nodeEmbSize, B, 1); % [Emb, B, 1]
    scores = sum(keys .* reshaped_query, 1); % sum over Emb dim -> [1, B, N]
    scores = scores / scaleFactor;

    % Add layer normalization to scores (optional stabilization)
    scores_mean = mean(scores, 3); % Mean over N dim
    scores_std = std(scores, 0, 3) + 1e-5; % Std dev over N dim
    scores = (scores - scores_mean) ./ scores_std;

    % Apply mask
    scores(mask) = -inf;

    % --- Calculate Probabilities ---
    scores_ = stripdims(scores); % [1, B, N]
    % Use temperature parameter to control softmax sharpness
    scores_scaled = scores_ / temperature;
    probs = softmax(scores_scaled, "DataFormat", "TBC"); % Softmax over N -> [1, B, N]

    % --- Sample Next City ---
    idx = zeros(B, 1, 'uint32'); % [B, 1] indices (1 to N)
    probs_gathered = gather(extractdata(probs));
    scores_gathered = gather(extractdata(scores));
    mask_gathered = mask; % Convert mask dlarray

    for i = 1:B % Loop through batch items
        current_probs_raw = squeeze(probs_gathered(1, i, :));
        p = max(0, current_probs_raw);
        p_sum = sum(p);
        if p_sum > 1e-8 % Check sum before division
            p = p / p_sum; % Normalize
        else
            p(:) = 1/N; % Fallback to uniform if sum is too small
            % This case might indicate all scores were -inf or numerically zero
        end

        if any(isnan(p)) || sum(p) < 1e-6 % Fallback
            current_mask = squeeze(mask_gathered(1, i, :));
            valid_indices = find(~current_mask);
            if isempty(valid_indices)
                current_scores = squeeze(scores_gathered(1, i, :));
                [~, max_idx_val] = max(current_scores);
                idx(i) = max_idx_val; % Should technically not happen if N steps
            else
                current_scores_valid = squeeze(scores_gathered(1, i, valid_indices));
                [~, max_idx_local] = max(current_scores_valid);
                idx(i) = valid_indices(max_idx_local);
            end
        else
            idx(i) = randsample(N, 1, true, p(:));
        end
    end % End batch loop

    % --- Store Results ---
    tours(:,t) = idx;
    linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
    logProbs(:,t) = log(probs(linearIndicesProbs) + 1e-10);

    % --- Calculate Entropy ---
    stepEntropy = -sum(probs .* log(probs + 1e-10), 3); % -> [1, B, 1]
    entropy_steps(1, :, t) = stepEntropy;

    % --- Update Mask ---
    linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(idx));
    mask(linearIndicesMask) = true;

    % --- Update Context for Next Step ---
    linearIndicesEmb = sub2ind([B, N], (1:B)', double(idx)); % Index into dims 2 & 3
    featuresSelected = nodeEmbeddings(:, linearIndicesEmb); % -> [Emb, B]
    currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT'); % [Emb, B, 1]

end % End step loop

entropy = reshape(sum(entropy_steps, 3), B, 1); % Final entropy [B, 1]
end

function tours = greedyDecode_GNN(net, dlX, numGNNLayers, ~, nodeEmbSize, temperature)
% dlX: [C=2, B=1, T=N]
B = size(dlX, 2); % Should be 1
N = size(dlX, 3);

% 1. --- Encoder: GNN Forward Pass ---
nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize); % [Emb, 1, N]

% --- Extract Weights ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys ---
keys = fullyconnect(nodeEmbeddings, Wk, Bk); % [Emb, 1, N]

% --- Initialize ---
mask = false(1, B, N); % [1, 1, N]
tours = zeros(B, N, 'uint32'); % [1, N]
meanEmbedding = mean(nodeEmbeddings, 3); % [Emb, 1, 1]
currentContext = meanEmbedding;
scaleFactor = sqrt(single(nodeEmbSize));

% For Monte Carlo sampling - introduce randomness
inferenceTempModifier = 0.05; % Adjusts temperature for inference (sharper distribution)
actualInferenceTemp = temperature * inferenceTempModifier;
useRandom = true;   % Set to true for sampling, false for pure greedy


% 2. --- "Decoder" Loop ---
for t = 1:N
    % --- Generate Query ---
    query_input = reshape(currentContext, nodeEmbSize, B); % [Emb, 1]
    query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB"); % [Emb, 1]

    % --- Attention Scores ---
    reshaped_query = reshape(query, nodeEmbSize, B, 1); % [Emb, 1, 1]
    scores = sum(keys .* reshaped_query, 1); % -> [1, 1, N]
    scores = scores / scaleFactor;

    % Layer normalization for scores
    scores_mean = mean(scores, 3);
    scores_std = std(scores, 0, 3) + 1e-5;
    scores = (scores - scores_mean) ./ scores_std;

    % Apply mask
    scores(mask) = -inf;

    % --- Select Next City (with temperature) ---
    if useRandom
        % Convert scores to probabilities with temperature
        scores_ = stripdims(scores);
        scores_temp = scores_ / actualInferenceTemp; % Apply temperature
        probs = softmax(scores_temp, "DataFormat", "TBC");

        % Sample from distribution
        probs_gathered = gather(extractdata(probs));
        current_probs = squeeze(probs_gathered(1, 1, :));

        % Ensure valid probabilities for sampling
        current_probs = max(0, current_probs);
        sum_probs = sum(current_probs);
        if sum_probs > 1e-8
            current_probs = current_probs / sum_probs;
        else
            current_probs(:) = 1/N;
        end

        % Sample from the distribution
        idx = randsample(N, 1, true, current_probs);
    else
        % Original greedy selection
        [~, idx] = max(scores, [], 3);
        idx = squeeze(idx);
        idx = uint32(extractdata(idx));
    end

    % --- Store Results & Update Mask ---
    tours(:,t) = idx;
    linearIndicesMask = sub2ind(size(mask), 1, 1, double(idx));
    mask(linearIndicesMask) = true;

    % --- Update Context ---
    linearIndicesEmb = sub2ind([B, N], 1, double(idx));
    featuresSelected = nodeEmbeddings(:, linearIndicesEmb); % -> [Emb, 1]
    currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT'); % [Emb, 1, 1]
end % End step loop
end
%% --- NEW GNN Actor Loss ---
function [gradients, loss] = actorLoss_GNN(net, dlX, adv, tours, entropyCoeff, numGNNLayers, nCities, nodeEmbSize, temperature)
% Re-calculate logProbs and entropy using GNN model inside dlfeval
[logProbs, entropy] = calculateLogProbsAndEntropyForTours_GNN(net, dlX, tours, numGNNLayers, nCities, nodeEmbSize, temperature);

% Policy Gradient Term
sumLogProbs = sum(logProbs, 2); % Shape: [B, 1]
adv = dlarray(adv(:));
policyLoss = -mean(sumLogProbs .* adv);

% Entropy Bonus Term
entropyLoss = -mean(entropy);

% Combined Loss
loss = policyLoss + entropyCoeff * entropyLoss;

gradients = dlgradient(loss, net.Learnables);
end


%% --- NEW GNN LogProb/Entropy Calculation Helper ---
function [logProbs, entropy] = calculateLogProbsAndEntropyForTours_GNN(net, dlX, tours, numGNNLayers, nCities, nodeEmbSize, temperature) %#ok<INUSD>
% Mirrors actorStep_GNN but uses provided tours

B = size(dlX, 2);
N = size(dlX, 3);

% 1. --- Encoder: GNN Forward Pass ---
nodeEmbeddings = gnnForward(net, dlX, numGNNLayers, N, nodeEmbSize); % [Emb, B, N]

% --- Extract Weights ---
Wk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bk = net.Learnables{strcmp(net.Learnables.Layer,'fcKey') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};
Wq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Weights'), 'Value'}{1};
Bq = net.Learnables{strcmp(net.Learnables.Layer,'fcQuery') & strcmp(net.Learnables.Parameter,'Bias'), 'Value'}{1};

% --- Pre-calculate Keys ---
keys = fullyconnect(nodeEmbeddings, Wk, Bk); % [Emb, B, N]

% --- Initialize ---
mask = false(1, B, N);
logProbs = zeros(B, N, 'like', dlX);
entropy_steps = zeros(1, B, N, 'like', dlX);
meanEmbedding = mean(nodeEmbeddings, 3);
currentContext = meanEmbedding;
scaleFactor = sqrt(single(nodeEmbSize));

% 2. --- Loop through steps, using provided tours(t) ---
for t = 1:N
    % --- Generate Query ---
    query_input = reshape(currentContext, nodeEmbSize, B);
    query = fullyconnect(query_input, Wq, Bq, "DataFormat", "CB");

    % --- Attention Scores ---
    reshaped_query = reshape(query, nodeEmbSize, B, 1);
    scores = sum(keys .* reshaped_query, 1); % -> [1, B, N]
    scores = scores / scaleFactor;

    % Layer normalization for scores
    scores_mean = mean(scores, 3);
    scores_std = std(scores, 0, 3) + 1e-5;
    scores = (scores - scores_mean) ./ scores_std;

    % Apply mask
    scores(mask) = -inf;

    % --- Calculate Probabilities ---
    scores_ = stripdims(scores);
    % Add a temperature parameter to control softmax sharpness
    scores_scaled = scores_ / temperature;
    probs = softmax(scores_scaled, "DataFormat", "TBC"); % Softmax over N -> [1, B, N]

    % --- Use provided tour index ---
    idx = tours(:, t); % idx is [B, 1]

    % --- Store Log Probability ---
    linearIndicesProbs = sub2ind(size(probs), ones(B, 1), (1:B)', double(idx));
    logProbs(:,t) = log(probs(linearIndicesProbs) + 1e-10); % logProbs is [B, N]

    % --- Calculate Entropy ---
    stepEntropy = -sum(probs .* log(probs + 1e-10), 3); % -> [1, B, 1]
    entropy_steps(1, :, t) = stepEntropy;

    % --- Update Mask ---
    linearIndicesMask = sub2ind(size(mask), ones(B,1), (1:B)', double(idx));
    mask(linearIndicesMask) = true;

    % --- Update Context ---
    linearIndicesEmb = sub2ind([B, N], (1:B)', double(idx));
    featuresSelected = nodeEmbeddings(:, linearIndicesEmb); % -> [Emb, B]
    currentContext = dlarray(reshape(featuresSelected, nodeEmbSize, B, 1), 'CBT'); % [Emb, B, 1]

end % End step loop

entropy = reshape(sum(entropy_steps, 3), B, 1); % Final entropy [B, 1]
end


%% --- Other Helper Functions ---

% Critic Loss
function [gradients, loss] = criticLoss(net, criticInputSequence, rewards)
% criticInputSequence should be [Emb, Batch, NCities] 'CBT' from GNN
baselinePredicted = forward(net, criticInputSequence); % Output [1, B, 1] 'CBT'
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
% N = size(coords,2);
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

function gradients = checkAndFixGradients(gradients)
for i = 1:height(gradients)
    if ~isempty(gradients.Value{i})
        if any(isnan(gradients.Value{i}(:)))
            % warning('NaN gradients detected and replaced with zeros');
            gradients.Value{i}(isnan(gradients.Value{i})) = 0;
        end
        if any(isinf(gradients.Value{i}(:)))
            % warning('Inf gradients detected and replaced with zeros');
            gradients.Value{i}(isinf(gradients.Value{i})) = 0;
        end
    end
end
end

%% New Pre-training Function
function [actorNet, avgGradA, avgSqGradA] = preTrainWithTwoOpt(actorNet, numPreTrainEpochs, preTrainBatchSize, nCities, nodeEmbSize, numGNNLayers, preTrainLR, temperature)
fprintf('Starting actor pre-training with 2-opt solutions for %d epochs...\n', numPreTrainEpochs);

% Initialize Adam optimizer states
avgGradA = [];
avgSqGradA = [];

% Track pre-training loss
preTrainLosses = zeros(1, numPreTrainEpochs);

% CHANGE: Use a more conservative learning rate to start
initialLR = preTrainLR * 0.1;
currentLR = initialLR;

% CHANGE: Use higher temperature during pre-training for better stability
preTrainTemp = max(1.0, temperature);

% CHANGE: Add gradient clipping threshold
gradClipThreshold = 1.0;

for epoch = 1:numPreTrainEpochs
    % Generate random TSP instances
    coords = rand(preTrainBatchSize, nCities, 2, 'single');

    % Calculate 2-opt solutions for each instance
    optimalTours = zeros(preTrainBatchSize, nCities, 'uint32');
    for i = 1:preTrainBatchSize
        coordsMatrix = squeeze(double(coords(i,:,:)));
        startCity = randi(nCities);
        [tourHeuristic, ~] = solveTSP_2opt(coordsMatrix, startCity);
        optimalTours(i,:) = uint32(tourHeuristic);
    end

    % Prepare input for actor network
    X = permute(coords, [3 1 2]); % [C=2, B, T=N]
    dlX = dlarray(X, 'CBT');

    % CHANGE: Add gradient checking with try-catch
    try
        [gradients, loss] = dlfeval(@supervisedActorLoss, actorNet, dlX, optimalTours, numGNNLayers, nCities, nodeEmbSize, preTrainTemp);

        % CHANGE: Improved gradient clipping
        gradients = thresholdL2Norm(gradients, gradClipThreshold);

        % Check for valid gradients before updating
        if isValidGradients(gradients)
            [actorNet, avgGradA, avgSqGradA] = adamupdate(actorNet, gradients, avgGradA, avgSqGradA, epoch, currentLR);
            preTrainLosses(epoch) = double(gather(loss));
        else
            fprintf('Epoch %d: Invalid gradients detected, skipping update\n', epoch);
            if epoch > 1
                preTrainLosses(epoch) = preTrainLosses(epoch-1);
            else
                preTrainLosses(epoch) = NaN;
            end

            % CHANGE: Reduce learning rate when gradients are problematic
            currentLR = currentLR * 0.8;
            fprintf('Reducing learning rate to %.6f\n', currentLR);
        end
    catch e
        fprintf('Error in epoch %d: %s\n', epoch, e.message);
        if epoch > 1
            preTrainLosses(epoch) = preTrainLosses(epoch-1);
        else
            preTrainLosses(epoch) = NaN;
        end

        % CHANGE: Reduce learning rate on error
        currentLR = currentLR * 0.5;
        fprintf('Reducing learning rate to %.6f\n', currentLR);
    end

    if mod(epoch, 5) == 0 || epoch == 1
        fprintf('Pre-training Epoch %d/%d, Loss: %.4f, LR: %.6f\n',...
            epoch, numPreTrainEpochs, preTrainLosses(epoch), currentLR);
    end

    % CHANGE: Learning rate schedule - increase gradually if training is stable
    if epoch > 10 && mod(epoch, 10) == 0 && isfinite(preTrainLosses(epoch))
        % If last 5 epochs show decreasing loss, slightly increase LR
        if all(diff(preTrainLosses(epoch-4:epoch)) <= 0)
            currentLR = min(currentLR * 1.2, preTrainLR);
            fprintf('Increasing learning rate to %.6f\n', currentLR);
        end
    end
end

% Plot pre-training loss curve (using only valid losses)
validLosses = preTrainLosses(isfinite(preTrainLosses));
validEpochs = find(isfinite(preTrainLosses));

figure;
plot(validEpochs, validLosses, '-o', 'MarkerSize', 4);
title('Actor Pre-training with 2-opt Solutions');
xlabel('Epoch'); ylabel('Cross-Entropy Loss'); grid on;

fprintf('Pre-training completed. Final loss: %.4f\n', preTrainLosses(end));
end

% CHANGE: Add function to check gradient validity
function valid = isValidGradients(gradients)
valid = true;
for i = 1:height(gradients)
    if ~isempty(gradients.Value{i})
        if any(isnan(gradients.Value{i}(:))) || any(isinf(gradients.Value{i}(:)))
            valid = false;
            return;
        end
    end
end
end

function [gradients, loss] = supervisedActorLoss(net, dlX, targetTours, numGNNLayers, nCities, nodeEmbSize, temperature)
% CHANGE: Add numerical stability measures


% Calculate log probabilities for supervised training
[logProbs, ~] = calculateLogProbsAndEntropyForTours_GNN(net, dlX, targetTours, numGNNLayers, nCities, nodeEmbSize, temperature);

% Check for NaN logProbs and replace with large negative values
nanMask = isnan(logProbs);
if any(nanMask(:))
    % CHANGE: Replace NaN with very negative but finite log probabilities
    logProbs(nanMask) = -100;
    fprintf('Warning: NaN values in logProbs replaced\n');
end

% Sum log probabilities across tour steps
sumLogProbs = sum(logProbs, 2); % [B, 1]

% CHANGE: Apply gradient stop for very negative values to avoid excessive penalties
extremeNegMask = sumLogProbs < -50;
if any(extremeNegMask)
    % Add small detached gradient for extreme negative values
    detached = extractdata(sumLogProbs);
    detached(extremeNegMask) = -50;
    sumLogProbs = dlarray(detached, 'CB');
end

% CHANGE: Improved loss calculation with clipping
clippedLogProbs = max(sumLogProbs, -50); % Avoid extremely negative values
loss = -mean(clippedLogProbs);

% Compute gradients
gradients = dlgradient(loss, net.Learnables);

% CHANGE: Better gradient handling
for i = 1:height(gradients)
    if ~isempty(gradients.Value{i})
        grad = gradients.Value{i};

        % Handle NaN/Inf values
        if any(isnan(grad(:))) || any(isinf(grad(:)))
            % fprintf('Warning: NaN/Inf gradients in %s/%s\n', gradients.Layer{i}, gradients.Parameter{i});

            % Extract data, clean, and reconstruct
            gradData = extractdata(grad);
            gradData(isnan(gradData) | isinf(gradData)) = 0;

            % Extra stability: clip extreme values
            maxMagnitude = 10.0;
            gradData = max(min(gradData, maxMagnitude), -maxMagnitude);

            gradients.Value{i} = dlarray(gradData);
        end
    end
end

% Supervised Loss Function for Pre-training

% Calculate log probabilities for supervised training
[logProbs, ~] = calculateLogProbsAndEntropyForTours_GNN(net, dlX, targetTours, numGNNLayers, nCities, nodeEmbSize, temperature);

% Cross-entropy loss: maximize log probability of optimal tours
sumLogProbs = sum(logProbs, 2); % Sum log probabilities across tour steps [B, 1]

% Instead of replacing NaNs with arbitrary values, use valid samples only
validSamples = ~isnan(sumLogProbs);

if any(validSamples)
    % Only use valid samples for loss calculation
    validLogProbs = sumLogProbs(validSamples);
    loss = -mean(validLogProbs); % Average negative log probability of valid samples
else
    % If all samples are invalid, return zero loss but with gradient tracking
    loss = dlarray(0);
    fprintf('Warning: All samples in batch contained NaN values.\n');
end

% Compute gradients
gradients = dlgradient(loss, net.Learnables);

% Handle NaN gradients safely without arbitrary values
for i = 1:numel(gradients.Value)
    if ~isempty(gradients.Value{i})
        % Get the current gradient tensor
        grad = gradients.Value{i};

        if any(isnan(grad(:)))
            % Create a mask of valid gradient values
            validMask = ~isnan(grad);

            if any(validMask(:))
                % Use only valid parts of the gradient
                grad = extractdata(grad);
                grad(~validMask) = 0; % Zero out NaN values
                gradients.Value{i} = dlarray(grad);
            else
                % If all gradient values are NaN, use zeros
                gradients.Value{i} = dlarray(zeros(size(grad), 'like', grad));
                % fprintf('Warning: All gradient values were NaN for parameter %s in layer %s.\n', ...
                %     gradients.Parameter{i}, gradients.Layer{i});
            end
        end
    end
end
end