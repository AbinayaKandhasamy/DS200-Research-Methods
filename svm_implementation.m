clc;
clearvars;
close all;

%% PARAMETERS
path = '/home/abinaya/Sem1/MLSP/Assigment2/4/Data/emotion_classification/';
K = 20; % pca dimension

%% LOAD TRAINING IMAGES AND THE GROUND TRUTH
train_images = dir([path 'train/*.gif']);
N = length(train_images);
label_vec = zeros(length(train_images),1);
train_img_vectors = [];

for i=1:length(train_images)
    if strfind(train_images(i).name,'happy')
        label_vec(i) = 1;
    else if strfind(train_images(i).name,'sad')
            label_vec(i) = 2;
        end
    end
    img = imread([path 'train/' train_images(i).name]);
    [m,n] = size(img);
    D = m*n;
    img1 = reshape(img,D,1);
    train_img_vectors = [train_img_vectors img1];
end

train_img_vectors1 = double(train_img_vectors);

%% LOAD TEST IMAGES AND THE GROUND TRUTH
test_images = dir([path 'test/*.gif']);
ground_truth = zeros(length(test_images),1);
test_img_vectors = [];

for i=1:length(test_images)
    if strfind(test_images(i).name,'happy')
        ground_truth(i) = 1;
    else if strfind(test_images(i).name,'sad')
            ground_truth(i) = 2;
        end
    end
    img = imread([path 'test/' test_images(i).name]);
    [m,n] = size(img);
    D = m*n;
    img1 = reshape(img,D,1);
    test_img_vectors = [test_img_vectors img1];
end
test_img_vectors1 = double(test_img_vectors);
%% Perform high dimensional PCA

sample_mean=(mean(train_img_vectors1,2));
sample_mean1 = repmat(sample_mean,1,N);
X = train_img_vectors1 - sample_mean1;


temp1=(1/N)*X'*(X);

[Vi, lambdai]=eig(temp1);

lambdas = sum(lambdai);
[lambda_sorted,sort_index]=sort(lambdas,'descend');
Vi_sorted = Vi(:,sort_index);

Ui=X*Vi_sorted;

for i=1:N
    Ui_normalized(:,i) = Ui(:,i)/((abs(lambda_sorted(i))*N)^0.5);
end

W_pca = Ui_normalized(:,1:K);
Y = W_pca'*X;

%% SVM Classifier
SVMSTRUCT = svmtrain(Y',label_vec);

%% TESTING
% PCA
 test0 = test_img_vectors1 - repmat(sample_mean,1,length(test_images));
 test = W_pca'*test0;
% classifier_result = svmclassify(SVMSTRUCT, test');
% % kernal functions : linear, quadratic, polynomial, rbf
% 
% 
% %% Accuracy
% accuracy = (sum(classifier_result == ground_truth)/ length(ground_truth))*100;
% display(accuracy);

%% Choice of K for PCA

accuracy_k = zeros(1,20);
for k = 1:20
    W_pca = Ui_normalized(:,1:k);
    Y = W_pca'*X;
    SVMSTRUCT = svmtrain(Y',label_vec);
    test = W_pca'*test0;
    accuracy_k(k) = (sum(svmclassify(SVMSTRUCT, test') == ground_truth)/ length(ground_truth))*100;
end

%% Change with respect to epsilon and C
epsilon_val = [0.3 0.2 0.1 0.01 0.001 0.0001 0.00001 0.000001];
c_val = [0.002 0.0039 0.0078 0.0156 0.0312 0.0625 0.125 0.25 0.5 1 2 8 16 32 64 128];
c_test_acc = zeros(1, length(c_val));
c_tr_acc = zeros(1, length(c_val));
e_test_acc = zeros(1, length(epsilon_val));
e_tr_acc = zeros(1, length(epsilon_val));

for e = 1: length(c_val)
    SVMSTRUCT = svmtrain(Y',label_vec,'kernel_function', 'rbf','boxconstraint',c_val(e));
    classifier_result = svmclassify(SVMSTRUCT, test');
    c_test_acc(e) = (sum(classifier_result == ground_truth)/ length(ground_truth))*100;
    c_tr_acc(e) = (sum(svmclassify(SVMSTRUCT, Y') == label_vec)/ length(label_vec))*100;
end

for e = 1: length(epsilon_val)
    SVMSTRUCT = svmtrain(Y',label_vec,'kernel_function', 'rbf','tolkkt',epsilon_val(e));
    classifier_result = svmclassify(SVMSTRUCT, test');
    e_test_acc(e) = (sum(classifier_result == ground_truth)/ length(ground_truth))*100;
    e_tr_acc(e) = (sum(svmclassify(SVMSTRUCT, Y') == label_vec)/ length(label_vec))*100;
end

%% Change in Kernal Function

%Linear Kernal
SVMSTRUCT_l = svmtrain(Y',label_vec,'kernel_function', 'linear');
SVMSTRUCT_q = svmtrain(Y',label_vec,'kernel_function', 'quadratic');
SVMSTRUCT_p = svmtrain(Y',label_vec,'kernel_function', 'polynomial');
SVMSTRUCT_r = svmtrain(Y',label_vec,'kernel_function', 'rbf');

Accuracy_linear = (sum(svmclassify(SVMSTRUCT_l, test') == ground_truth)/ length(ground_truth))*100;
Accuracy_quadratic = (sum(svmclassify(SVMSTRUCT_q, test') == ground_truth)/ length(ground_truth))*100;
Accuracy_polynomial = (sum(svmclassify(SVMSTRUCT_p, test') == ground_truth)/ length(ground_truth))*100;
Accuracy_rbf = (sum(svmclassify(SVMSTRUCT_r, test') == ground_truth)/ length(ground_truth))*100;
display(Accuracy_linear);display(Accuracy_quadratic);display(Accuracy_polynomial);display(Accuracy_rbf);


%% LDA

m1 = mean(Y(:,label_vec==1),2);
m2 = mean(Y(:,label_vec==2),2);

%Within Class Variance
SW = zeros(K,K);
for i=1:N
    if label_vec(i)==1
        temp = Y(:,i) - m1;
    end
    if label_vec(i)==2
        temp = Y(:,i) - m2;
    end
    SW = SW + (temp*temp');
end

W_lda=SW\(m2-m1);
projected = W_lda'*Y;
te_projected = W_lda'*test;

SVMSTRUCT_lda = svmtrain(projected',label_vec,'kernel_function', 'linear');
Accuracy_lda = (sum(svmclassify(SVMSTRUCT_lda, te_projected') == ground_truth)/ length(ground_truth))*100;