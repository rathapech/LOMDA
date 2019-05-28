clear
clc

DSmat='dataset/DSmat.txt';
RSmat='dataset/RSmat.txt';
RDmat='dataset/RDmat.txt';
DS=load(DSmat);
RS=load(RSmat);
RD=load(RDmat);

Adj=[RS,RD;RD',DS];

% disease
% 624:Breast neoplasms
% 633:Hepatocellular carcinoma
% 636:Renal cell carcinoma
% 638:Squamous cell carcinoma
% 662:Colorectal neoplasms
% 706:Glioblastoma
% 721:Heart failure
% 764:Acute myeloid leukemia
% 781:Lung neoplasms
% 898:Melanoma
% 847:Ovarian neoplasms
% 849:Pancreatic neoplasms
% 866:Prostatic neoplasms
% 898:Stomach neoplasms
% 907:Urinary bladder neoplasms

disease_id = 636; % Change this variable corresponding to different disease

index=find(Adj(disease_id,1:577));

indices = crossvalind('Kfold', index, 5);

for k=1:1                                    
    test = (indices == k);                    
    train = ~test;                        
    AdjTraining=Adj;
    for i=1:length(train)
        if train(i)==0
            AdjTraining(disease_id,i)=0;
            AdjTraining(i,disease_id)=0;
        end
    end   

    AdjProb=Adj-AdjTraining;
    
    alpha_A = 0.01;
    S = (1/alpha_A*eye(size(AdjTraining'*AdjTraining)) + ...
                    AdjTraining' * AdjTraining)\AdjTraining' * AdjTraining;
    probMatrix = AdjTraining * S;
    

    %calculate the AUC score
    index1=find(tril(AdjProb,-1));
    weight1=probMatrix(index1);
    index2=find(tril(~Adj,-1));
    weight2=probMatrix(index2); 
    labels=[];
    scores=[];
    labels(1:length(weight1))=1;
    labels(end+1:end+length(weight2))=0;
    scores(1:length(weight1))=weight1;
    scores(end+1:end+length(weight2))=weight2;
    [X,Y,T,AUC] = perfcurve(labels,scores,1);
end

disp(['AUC is : ' num2str(AUC)]);

