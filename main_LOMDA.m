clear all;
clc;
close all;


num_simulation = 2; % Number of independent simulations, for nuclear receptors (4) 
       % we recommond to set the number of simulation up to 10 or more 
       % since the performance is not so stable because the network is too
       % small
       
K = 5; % K-fold cross validation

% Initialize parameter alpha 
alpha_A = 0.01;
alpha_MD = 0.001;

DSmat='dataset/DSmat.txt';
RSmat='dataset/RSmat.txt';
MDmat='dataset/RDmat.txt';

DS=load(DSmat); % Disease similarity
RS=load(RSmat); % MiRNA similarity
MD=load(MDmat); % Known associations

% Allocation the memory for some variables 
aucLORun_A = zeros(num_simulation,1);
auprLORun_A = zeros(num_simulation,1);

aucLORun_MD = zeros(num_simulation,1);
auprLORun_MD = zeros(num_simulation,1);

for r = 1 : num_simulation % Number of independent simulations 
    disp('===========================================================');
    disp(['============= r = ' num2str(r) ' ========== of ' num2str(num_simulation) ' =======']);
    disp('===========================================================');
    fprintf('\n');
    
    y = MD;
    crossval_idx = crossvalind('Kfold',y(:),K); % K-fold division 
    
    % Allocation the memory for some variables 
    aucLO_A = zeros(K,1);
    auprLO_A = zeros(K,1);
    
    aucLO_MD = zeros(K,1);
    auprLO_MD = zeros(K,1);
    
    for i = 1 : K % each fold at a time
        disp(['--- Run ' num2str(r) ' of ' num2str(num_simulation) ', k ' num2str(i) ' of ' num2str(K) ' ---']);

        train_idx = find(crossval_idx~=i);
        test_idx  = find(crossval_idx==i);

        y_train = y;
        y_train(test_idx) = 0;
        train = y_train;
        test = y - train;
        
        yy=y;
        yy(yy==0)=-1;

        temp=[RS,train;train',DS]; % Integration matrix \mathbf{A}

        loMatrixA = (1/alpha_A*eye(size(temp'*temp)) + ...
                    temp' * temp)\temp' * temp; % Computing the weighting matrix
        loMatrixA = temp * loMatrixA;
        newLowMatA = loMatrixA(size(RS,1)+1:end,1:size(MD',2))';   
        statsLO_A = evaluate_performance(newLowMatA(test_idx),yy(test_idx),'classification');
        aucLO_A(i) = statsLO_A.auc;
        auprLO_A(i) = statsLO_A.aupr;
        
        % Working with only known association 
        loMatrixMD = (1/alpha_MD*eye(size(train'*train)) + train' * train)\train' * train;
        loMatrixMD = train * loMatrixMD; 
        statsLO_MD = evaluate_performance(loMatrixMD(test_idx),yy(test_idx),'classification');
        aucLO_MD(i) = statsLO_MD.auc;
        auprLO_MD(i) = statsLO_MD.aupr;
        
        fprintf('\n');
    end
    aucLORun_A(r) = mean(aucLO_A);
    auprLORun_A(r) = mean(auprLO_A);
    
    aucLORun_MD(r) = mean(aucLO_MD);
    auprLORun_MD(r) = mean(auprLO_MD);
    
    fprintf('\n');

end

% Printing results 
disp(['The AUC from LOMDA-A: ' num2str(mean(aucLORun_A))]);
disp(['The AUPR from LOMDA-A: ' num2str(mean(auprLORun_A))]);
fprintf('\n');


disp(['The AUC from LOMDA-MD: ' num2str(mean(aucLORun_MD))]);
disp(['The AUPR from LOMDa-MD: ' num2str(mean(auprLORun_MD))]);
fprintf('\n');
    

    

    


