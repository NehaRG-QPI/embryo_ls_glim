%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT TO COMBINE SLICE LEVEL PREDICTIONS OF IBM INTO EMBRYO LEVEL CLASS
% SCRIPT DEVELOPED BY: NEHA GOSWAMI, UIUC, IL, USA
% PUBLICATION: Goswami, Neha, et al. "Machine learning assisted health 
% viability assay for mouse embryos with artificial confocal microscopy 
% (ACM)." bioRxiv (2023): 2023-07.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
pth='your system path to embryo_ls_glim-main/example_data/healthy (or sick, or sparse_prediction)/overlapped'; % path to IBM predictions
filename=sprintf('check.csv');% IBM prediction csv file name
T=readtable(fullfile(pth,filename));
T2=T;
for i=1:size(T2,1)
    name(i,1)=extractBefore(T2{i,2},'_z'); % extracting names of individual embryos
    category(i,1)=T2(i,4); % extracting actual class
end
T.name=name;
T.category=category;
unique_names=unique(T.name);
error_prob=zeros(numel(unique_names),1);
total_prob=zeros(numel(unique_names),1);
true_category=zeros(numel(unique_names),1);
mode_pred=zeros(numel(unique_names),1);

for i=1:numel(unique_names)
    good=0;
    bad=0;
    name2=unique_names(i);
    idx=find(strcmp(T.name,name2)); % collecting all z-slices of inidividual embryo
    true_category(i)=unique(T.actual(idx)); % collecting actual class information
    mode_pred(i)=mode(T.predicted(idx)); % max-voting of prediction
    pred=T.predicted(idx);
    gt=T.actual(idx);
    prob=T.probability(idx);
    for j=1:numel(idx)
        if pred(j)==gt(j)
            total_prob(i)=total_prob(i)+prob(j); % correct probability
            good=good+1;
        else
            error_prob(i)=error_prob(i)+prob(j); % error probability
            bad=bad+1;
        end


    end
    if good
        total_prob(i)=total_prob(i)/(good); % averaged over correct subset
    else
        total_prob(i)=0;
    end


    if bad
        error_prob(i)=error_prob(i)/(bad);% averaged over error subset
    else
        error_prob(i)=0;
    end

    total_prob((isnan(total_prob)))=0;
    error_prob((isnan(error_prob)))=0;
    weight_t(i)=100*good/numel(idx); % percentage correct z-slices
    weight_e(i)=100*bad/numel(idx); % percentage error z-slices
end

%  write embryo-wise results table
T4=table;
T4.name=unique_names;
T4.actual_class=true_category;
T4.most_predicted=mode_pred;
T4.probability=total_prob;
T4.error_probability=error_prob;
T4.percentage_mode_error=weight_e';
T4.percentage_mode_total=weight_t';
writetable(T4,fullfile(pth,'result.csv'));
