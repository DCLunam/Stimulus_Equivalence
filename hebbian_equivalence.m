%Angel Tovar - aetovar@unam.mx , eugenio.tovar@gmail.com
%See the PDF file "Hebbian Equivalence About and Instructions"
%for a detailed description of this Matlab script and examples.
%Last update: January 2022

%SYNTAX
%[W,Test_relatedness,W_epochs] = hebbian_equivalence(S,C,T,epochs,random,LTP_threshold,beta,W,Stest,Ttest)

%Input Arguments
%S - Matrix with sample stimuli
%C - Matrix with comparison stimuli
%T - Matrix with target responses
%epochs - integer with the number of training epochs
%random - set to 1 to determine training trials presented in random order, otherwise the training trials are presented in the order determined in the training matrices
%LTP_threshold - LTD/LTP threshold for weight adjustments. above threshold
%values strengthen the weight, below threshold values weaken the weight. In this simulator a “typical” thresholds may be set around 0.35. Increas this value to model learning disabilities.
%beta - learning rate for training trials. set between 0 and 1
%W - Weight matrix, declare the name of a previously trained W, or set to 0 to initialize a new simulation
%Stest - Matrix with sample stimuli for tests
%Ttest - Matrix with target stimuli for tests

%Output Arguments
%W - Weight matrix by the end of training trials
%Test_relatedness - Matrix with relatedness values for each test trial (in rows) evaluated after each training epoch (in columns)
% W_epochs - 3d matrix with all W matrices obtained after each training epoch. 

function [W,Test_relatedness,W_epochs] = hebbian_equivalence(S,C,T,epochs,random,LTP_threshold,beta,W,Stest,Ttest)

[number_trials , number_stimuli] = size(S);

%Create the network and initialize connection weights
neurons = number_stimuli;
if W > 0 %The user wants to use an existing weight matrix
    W = W;
else %create a new weight matrix, default is 0 values, but random values are also possible
    W = zeros(neurons);
end

all_matrices = [S,C,T];
C_select_vector = zeros(1,neurons);
V = 0.7:0.01:0.99;%This creates random numbers to provide stochasticity to the model at each training step

%First start the loop for epochs
for ep = 1 : epochs
%Ordering trials    
if random == 1
    r = randperm(size(all_matrices,1));
    all_matrices = all_matrices(r,:);
end
    S = all_matrices(:,1:number_stimuli);
    C = all_matrices(:,number_stimuli+1:end-number_stimuli);
    T = all_matrices(:,end-number_stimuli+1:end);
    
%Start loop for trials
for tr = 1: number_trials  %trials
    
    %SELECT THE COMPARISON
    %pick the stimuli that gets the strongest activation from the sample, if none, pick randomly
    rep_S = repmat(S(tr,:),neurons,1); %repeats the sample vector times the number of neurons to compute all possible inputs
    net_input = dot(W,rep_S');%Net input from sample to comparisons
    C_ratings = net_input.*C(tr,:);%Keeps only values of comparisons presented in this trial
    C_rating_max = max(C_ratings);
    if sum(C_ratings) <= 0
        C_index = find(C(tr,:)>0);%identify all possible comparisons
        C_available = find(C_ratings >= 0);%to discard negativelly related comparisons in past trials
        [idx_available,loc] = ismember(C_index,C_available);
        out = loc(idx_available);
        rC = randperm(size(out',1));
        C_select = out(rC(1));
    else %if Weights are non 0 or negative values, select the strongest, if more than one, pick randomnly
        idx_max = find(C_ratings == C_rating_max);
        rC = randperm(size(idx_max',1));
        C_select = idx_max(rC(1));
    end
    C_select_vector(C_select) = 1;

%Compute activation in the network    
    active_units = S(tr,:) + C_select_vector;%external activation
    rep_active = repmat(active_units,neurons,1);
    net_input_active_units = dot(W,rep_active');%Net input from sample and selected comparison
    int_act =  net_input_active_units ./ (1+net_input_active_units);%Activation function. Modified from original paper. This one is more suitable for modeling
    final_act = active_units + int_act;%External and spreading activation
    final_act = ((final_act<1).*final_act)+(final_act>=1);%limiting act values, avoids surpassing 1

%%
%LEARNING PROCESS  
%Coactivation matrix
    act_1 = repmat(final_act,neurons,1);
    act_2 = repmat(final_act',1,neurons);
    coactivation = (act_1 .*act_2);

%lambda and ltp/ltd threshold
    W(W < 0) = 0;% first find if there are negative W values and set them to 0
    LambdaLTP = ((coactivation > LTP_threshold) .* coactivation) - (W);
    for la=1:neurons
        LambdaLTP(la,la)  = 0;%sets diagonal to 0
    end
    
% Check if the selected comparison matches the target comparison. This
% combines benefits from both unsupervised and reinforcement learning
    if isequal(C_select_vector,T(tr,:))
        Train_Responses(tr,ep) = 1; %Registers a correct response
        beta_positiveP = (LambdaLTP>=0).*(LambdaLTP.*(beta .*(V(randi([1,numel(V)]))) ));
        beta_negativeP = (LambdaLTP<0).*(LambdaLTP.*(beta .*(V(randi([1,numel(V)]))))); 
    else
        Train_Responses(tr,ep) = 0;%Registers an incorrect response
        beta_positiveP = (LambdaLTP>=0).*(LambdaLTP.*(-beta .*(V(randi([1,numel(V)]))))); 
        beta_negativeP = (LambdaLTP<0).*(LambdaLTP.*(beta .*(V(randi([1,numel(V)]))))); 
    end

    beta_pn = beta_positiveP  +  beta_negativeP;    
    delta = (act_1.*act_2).*beta_pn;%Hebbian learning
    W = W+delta;%updating W
    W_epochs(:,:,ep) = W;%Saving all Ws
    C_select_vector = zeros(1,neurons);
end %ends trials

% TESTS for derived relations, during each epoch
[number_tests,~] = size(Stest);
for t = 1:number_tests
    Stest_idx = find(Stest(t,:) == 1);
    Ttest_idx = find(Ttest(t,:) == 1);
    Test_relatedness(t,ep) = W(Stest_idx,Ttest_idx);
end %ends test trials

end %ends epochs