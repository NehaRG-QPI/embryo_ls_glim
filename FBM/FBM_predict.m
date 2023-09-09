%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT TO EVALUATE FBM AND COMBINE PREDICTIONS TO EMBRYO
% LEVEL CLASS
%
% SCRIPT DEVELOPED BY: NEHA GOSWAMI, UIUC, IL, USA
% PUBLICATION: Goswami, Neha, et al. "Machine learning assisted health
% viability assay for mouse embryos with artificial confocal microscopy
% (ACM)." bioRxiv (2023): 2023-07.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters
clc;
clearvars -except count;
pthm='your system path to embryo_ls_glim-main/FBM'; % path where pretrained model is saved
pret_m='model_94202.mat';
pthd='your system path to embryo_ls_glim-main/example_data/healthy (or sick)';% path to load test data
test_e=readtable(fullfile(pthd,'var3.csv'));
m=load(fullfile(pthm,pret_m));
Mdl=m.Mdl;
%% predictions
[l,cl]=predict(Mdl,test_e);
Tw=table;
Tw.num=[0:1:numel(l)-1]';
Tw.predicted=l;
Tw.actual=test_e.new_num_cat;
Tw.probability=max(cl,[],2);
filename_p='check.csv';
filename_t='unknown.csv';
filename_r='result.csv';
writetable(Tw,fullfile(pthd,filename_p));
%% prepare gt
Tg=table;
Tg.name=test_e.name;
Tg.new_num_cat=test_e.new_num_cat;
writetable(Tg,fullfile(pthm,filename_t));
combine_predictions(pthd,pthd,filename_p,filename_t,filename_r);

%%
function combine_predictions(pth,ptht,filename_p,filename_t,filename_r)
T=readtable(fullfile(pth,filename_p));
T2=readtable(fullfile(ptht,filename_t),'ReadVariableNames',true);
for i=1:size(T2,1)

    name(i,1)=T2{i,1};
    category(i,1)=T2(i,2);
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
    idx=find(strcmp(T.name,name2));
    true_category(i)=unique(T.actual(idx));
    mode_pred(i)=mode(T.predicted(idx));
    pred=T.predicted(idx);
    gt=T.actual(idx);
    prob=T.probability(idx);
    for j=1:numel(idx)
        if pred(j)==gt(j)
            total_prob(i)=total_prob(i)+prob(j);
            good=good+1;
        else
            error_prob(i)=error_prob(i)+prob(j);
            bad=bad+1;
        end


    end
    if good
        total_prob(i)=total_prob(i)/good;
    else
        total_prob(i)=0;
    end


    if bad
        error_prob(i)=error_prob(i)/bad;
    else
        error_prob(i)=0;
    end

    total_prob((isnan(total_prob)))=0;
    error_prob((isnan(error_prob)))=0;
    r_t=numel(nonzeros(total_prob(i)));
    r_e=numel(nonzeros(error_prob(i)));
    weight_t(i)=100*good/numel(idx);
    weight_e(i)=100*bad/numel(idx);
end


T4=table;
T4.name=unique_names;
T4.actual_class=true_category;
T4.most_predicted=mode_pred;
T4.probability=total_prob;
T4.error_probability=error_prob;
T4.percentage_mode_error=weight_e';
T4.percentage_mode_correct=weight_t';
writetable(T4,fullfile(pth,filename_r));
end


