clear all
close all
clc

data = csvread('house_prices_data_training_data.csv',1,2);

data60Percent = data(1:12965 , :);
dataCV = data(12966:17288 , :);
dataFinal = data(17289:end , :);

% neededFeatures1Hyp1 = data60Percent(:,4:5).^4;
% neededFeatures2Hyp1 = data60Percent(:,11:12).^4;
% neededFeatures3Hyp1 = data60Percent(:,18:19).^4;
% neededFeatures1Hyp1CV = dataCV(:,4:5).^4;
% neededFeatures2Hyp1CV = dataCV(:,11:12).^4;
% neededFeatures3Hyp1CV = dataCV(:,18:19).^4;
% neededFeatures1Hyp1Final = dataFinal(:,4:5).^4;
% neededFeatures2Hyp1Final = dataFinal(:,11:12).^4;
% neededFeatures3Hyp1Final = dataFinal(:,18:19).^4;
% 
% neededFeatures1Hyp2 = data60Percent(:,2:3).^10;
% neededFeatures2Hyp2 = data60Percent(:,6).^10;
% neededFeatures3Hyp2 = data60Percent(:,9:10).^10;
% neededFeatures1Hyp2CV = dataCV(:,2:3).^10;
% neededFeatures2Hyp2CV = dataCV(:,6).^10;
% neededFeatures3Hyp2CV = dataCV(:,9:10).^10;
% neededFeatures1Hyp2Final = dataFinal(:,2:3).^10;
% neededFeatures2Hyp2Final = dataFinal(:,6).^10;
% neededFeatures3Hyp2Final = dataFinal(:,9:10).^10;
% 
% 
% neededFeatures1Hyp3 = data60Percent(:,15:17).^8;
% neededFeatures1Hyp3CV = dataCV(:,15:17).^8;
% neededFeatures1Hyp3Final = dataFinal(:,15:17).^8;
% 
% neededFeatures1Hyp4 = data60Percent(:,4).^2;
% neededFeatures2Hyp4 = data60Percent(:,10).^2;
% neededFeatures3Hyp4 = data60Percent(:,18).^2;
% neededFeatures4Hyp4 = data60Percent(:,6).^2;
% neededFeatures1Hyp4CV = dataCV(:,4).^2;
% neededFeatures2Hyp4CV = dataCV(:,10).^2;
% neededFeatures3Hyp4CV = dataCV(:,18).^2;
% neededFeatures4Hyp4CV = dataCV(:,6).^2;
% neededFeatures1Hyp4Final = dataFinal(:,4).^2;
% neededFeatures2Hyp4Final = dataFinal(:,10).^2;
% neededFeatures3Hyp4Final = dataFinal(:,18).^2;
% neededFeatures4Hyp4Final = dataFinal(:,6).^2;

AllFeatures = data60Percent(:,2:end);
AllFeaturesCV = dataCV(:,2:end);
AllFeaturesFinal = dataFinal(:,2:end);
AllFeaturesSquared = AllFeatures.^2;
AllFeaturesSquaredCV = AllFeaturesCV.^2;
AllFeaturesSquaredFinal = AllFeaturesFinal.^2;
allFeaturesCosined = log(AllFeatures);


results60Percent = data60Percent(:,1); %results of data
resultsCV = dataCV(:,1);
resultsCV(:)=(resultsCV(:)-mean((resultsCV(:))))./std(resultsCV(:));
resultsFinal = dataFinal(:,1);
resultsFinal(:)=(resultsFinal(:)-mean((resultsFinal(:))))./std(resultsFinal(:));
length = length(results60Percent);
lengthCV = size(resultsCV);
lengthFinal = size(resultsFinal);
alpha = 0.08;




%Hypothisis one
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp1 = data60Percent(:,4:5).^degree;
neededFeatures2Hyp1 = data60Percent(:,11:12).^degree;
neededFeatures3Hyp1 = data60Percent(:,18:19).^degree;  
    
X1 = [ones(length, 1),AllFeatures,neededFeatures1Hyp1,neededFeatures2Hyp1,neededFeatures3Hyp1];
n1 = size(X1(1,:));
[iterations1 , Js1 , theta1] = helperMLAss1(X1,alpha,results60Percent);
thetaVector1(:,counter) = theta1;
normalEquation1(:,counter) = normalEquation(X1, results60Percent);
figure(1)
plot(1: iterations1(2), Js1 , '-b');
title('Hypothesis one');
xlabel('Number of iterations');
ylabel('Error');
Js1(end)

neededFeatures1Hyp1CV = dataCV(:,4:5).^degree;
neededFeatures2Hyp1CV = dataCV(:,11:12).^degree;
neededFeatures3Hyp1CV = dataCV(:,18:19).^degree;

X1CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp1CV,neededFeatures2Hyp1CV,neededFeatures3Hyp1CV];
n = size(X1CV(1,:));
for w=2:n(2)
    if max(abs(X1CV(:,w)))~=0
    X1CV(:,w)=(X1CV(:,w)-mean((X1CV(:,w))))./std(X1CV(:,w));
    end
end
errorCVHyp1(counter) = ComputeCost(X1CV,resultsCV,thetaVector1(:,counter));
counter = counter + 1;
end

optDegree1 = find(errorCVHyp1 == min(errorCVHyp1))*2;

neededFeatures1Hyp1Final = dataFinal(:,4:5).^optDegree1;
neededFeatures2Hyp1Final = dataFinal(:,11:12).^optDegree1;
neededFeatures3Hyp1Final = dataFinal(:,18:19).^optDegree1;

X1Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp1Final,neededFeatures2Hyp1Final,neededFeatures3Hyp1Final];

n = size(X1Final(1,:));
for w=2:n(2)
    if max(abs(X1Final(:,w)))~=0
    X1Final(:,w)=(X1Final(:,w)-mean((X1Final(:,w))))./std(X1Final(:,w));
    end
end
errorFinalHyp1 = ComputeCost(X1Final,resultsFinal,thetaVector1(:,optDegree1/2));





%Hypothisis two
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp2 = data60Percent(:,2:3).^degree;
neededFeatures2Hyp2 = data60Percent(:,6).^degree;
neededFeatures3Hyp2 = data60Percent(:,9:10).^degree;
    
X2 = [ones(length, 1),AllFeatures,neededFeatures1Hyp2,neededFeatures2Hyp2,neededFeatures3Hyp2];
n2 = size(X2(1,:));
[iterations2 , Js2 , theta2] = helperMLAss1(X2,alpha,results60Percent);
thetaVector2(:,counter) = theta2;
normalEquation2(:,counter) = normalEquation(X2, results60Percent);
figure(2)
plot(1: iterations2(2), Js2 , '-b');
title('Hypothesis two');
xlabel('Number of iterations');
ylabel('Error');
Js2(end)

neededFeatures1Hyp2CV = dataCV(:,2:3).^degree;
neededFeatures2Hyp2CV = dataCV(:,6).^degree;
neededFeatures3Hyp2CV = dataCV(:,9:10).^degree;
X2CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp2CV,neededFeatures2Hyp2CV,neededFeatures3Hyp2CV];
n = size(X2CV(1,:));
for w=2:n(2)
    if max(abs(X2CV(:,w)))~=0
    X2CV(:,w)=(X2CV(:,w)-mean((X2CV(:,w))))./std(X2CV(:,w));
    end
end
errorCVHyp2(counter) = ComputeCost(X2CV,resultsCV,thetaVector2(:,counter));
counter = counter + 1;
end


optDegree2 = find(errorCVHyp2 == min(errorCVHyp2))*2;

neededFeatures1Hyp2Final = dataFinal(:,2:3).^optDegree2;
neededFeatures2Hyp2Final = dataFinal(:,6).^optDegree2;
neededFeatures3Hyp2Final = dataFinal(:,9:10).^optDegree2;

X2Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp2Final,neededFeatures2Hyp2Final,neededFeatures3Hyp2Final];

n = size(X2Final(1,:));
for w=2:n(2)
    if max(abs(X2Final(:,w)))~=0
    X2Final(:,w)=(X2Final(:,w)-mean((X2Final(:,w))))./std(X2Final(:,w));
    end
end
errorFinalHyp2 = ComputeCost(X2Final,resultsFinal,thetaVector2(:,optDegree2/2));






%Hypothisis three
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp3 = data60Percent(:,15:17).^degree;
    
X3 = [ones(length, 1),AllFeatures,neededFeatures1Hyp3];
n3 = size(X1(1,:));
[iterations3 , Js3 , theta3] = helperMLAss1(X3,alpha,results60Percent);
thetaVector3(:,counter) = theta3;
normalEquation3(:,counter) = normalEquation(X3, results60Percent);
figure(3)
plot(1: iterations3(2), Js3 , '-b');
title('Hypothesis three');
xlabel('Number of iterations');
ylabel('Error');
Js3(end)

neededFeatures1Hyp3CV = dataCV(:,15:17).^degree;
X3CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp3CV];
n = size(X3CV(1,:));
for w=2:n(2)
    if max(abs(X3CV(:,w)))~=0
    X3CV(:,w)=(X3CV(:,w)-mean((X3CV(:,w))))./std(X3CV(:,w));
    end
end
errorCVHyp3(counter) = ComputeCost(X3CV,resultsCV,thetaVector3(:,counter));
counter = counter + 1;
end

optDegree3 = find(errorCVHyp3 == min(errorCVHyp3))*2;

neededFeatures1Hyp3Final = dataFinal(:,15:17).^optDegree3;


X3Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp3Final];

n = size(X3Final(1,:));
for w=2:n(2)
    if max(abs(X3Final(:,w)))~=0
    X3Final(:,w)=(X3Final(:,w)-mean((X3Final(:,w))))./std(X3Final(:,w));
    end
end
errorFinalHyp3 = ComputeCost(X3Final,resultsFinal,thetaVector3(:,optDegree3/2));







% X3 = [ones(length, 1),AllFeaturesSquared,neededFeatures1Hyp3];
% n3 = size(X3(1,:));
% [iterations3 , Js3 , theta3] = helperMLAss1(X3,alpha,results60Percent);
% normalEquation3 = normalEquation(X3, results60Percent);
% figure(3)
% plot(1: iterations3(2), Js3 , '-b');
% title('Hypothesis three');
% xlabel('Number of iterations');
% ylabel('Error');
% Js3(end)
% X3CV = [ones(lengthCV(1), 1),AllFeaturesSquaredCV,neededFeatures1Hyp3CV];
% n = size(X3CV(1,:));
% for w=2:n(2)
%     if max(abs(X3CV(:,w)))~=0
%     X3CV(:,w)=(X3CV(:,w)-mean((X3CV(:,w))))./std(X3CV(:,w));
%     end
% end
% errorCVHyp333 = ComputeCost(X3CV,resultsCV,theta3);
% X3Final = [ones(lengthFinal(1), 1),AllFeaturesSquaredFinal,neededFeatures1Hyp3Final];
% n = size(X3Final(1,:));
% for w=2:n(2)
%     if max(abs(X3Final(:,w)))~=0
%     X3Final(:,w)=(X3Final(:,w)-mean((X3Final(:,w))))./std(X3Final(:,w));
%     end
% end
% errorFinalHyp333 = ComputeCost(X3Final,resultsFinal,theta3);






%Hypothisis four
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp4 = data60Percent(:,4).^degree;
neededFeatures2Hyp4 = data60Percent(:,10).^degree;
neededFeatures3Hyp4 = data60Percent(:,18).^degree;
neededFeatures4Hyp4 = data60Percent(:,6).^degree;
    
X4 = [ones(length, 1),AllFeatures,neededFeatures1Hyp4];
n4 = size(X1(1,:));
[iterations4 , Js4 , theta4] = helperMLAss1(X4,alpha,results60Percent);
thetaVector4(:,counter) = theta4;
normalEquation4(:,counter) = normalEquation(X4, results60Percent);
figure(4)
plot(1: iterations4(2), Js4 , '-b');
title('Hypothesis four');
xlabel('Number of iterations');
ylabel('Error');
Js4(end)

neededFeatures1Hyp4CV = dataCV(:,4).^degree;
neededFeatures2Hyp4CV = dataCV(:,10).^degree;
neededFeatures3Hyp4CV = dataCV(:,18).^degree;
neededFeatures4Hyp4CV = dataCV(:,6).^degree;
X4CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp4CV];
n = size(X4CV(1,:));
for w=2:n(2)
    if max(abs(X4CV(:,w)))~=0
    X4CV(:,w)=(X4CV(:,w)-mean((X4CV(:,w))))./std(X4CV(:,w));
    end
end
errorCVHyp4(counter) = ComputeCost(X4CV,resultsCV,thetaVector4(:,counter));
counter = counter + 1;
end

optDegree4 = find(errorCVHyp4 == min(errorCVHyp4))*2;

neededFeatures1Hyp4Final = dataFinal(:,4).^optDegree4;
neededFeatures2Hyp4Final = dataFinal(:,10).^optDegree4;
neededFeatures3Hyp4Final = dataFinal(:,18).^optDegree4;
neededFeatures4Hyp4Final = dataFinal(:,6).^optDegree4;


X4Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp4Final];

n = size(X4Final(1,:));
for w=2:n(2)
    if max(abs(X4Final(:,w)))~=0
    X4Final(:,w)=(X4Final(:,w)-mean((X4Final(:,w))))./std(X4Final(:,w));
    end
end
errorFinalHyp4 = ComputeCost(X4Final,resultsFinal,thetaVector4(:,optDegree4/2));













% X4 = [ones(length, 1),AllFeatures,neededFeatures1Hyp4,neededFeatures2Hyp4,neededFeatures3Hyp4,neededFeatures4Hyp4];
% n4 = size(X3(1,:));
% [iterations4 , Js4 , theta4] = helperMLAss1(X4,alpha,results60Percent);
% normalEquation4 = normalEquation(X4, results60Percent);
% figure(4)
% plot(1: iterations4(2), Js4 , '-b');
% title('Hypothesis four');
% xlabel('Number of iterations');
% ylabel('Error');
% Js4(end)
% X4CV = [ones(lengthCV(1), 1),AllFeaturesSquaredCV,neededFeatures1Hyp4CV,neededFeatures2Hyp4CV,neededFeatures3Hyp4CV,neededFeatures4Hyp4CV];
% n = size(X4CV(1,:));
% for w=2:n(2)
%     if max(abs(X4CV(:,w)))~=0
%     X4CV(:,w)=(X4CV(:,w)-mean((X4CV(:,w))))./std(X4CV(:,w));
%     end
% end
% errorCVHyp4 = ComputeCost(X4CV,resultsCV,theta4);
% X4Final = [ones(lengthFinal(1), 1),AllFeaturesSquaredFinal,neededFeatures1Hyp4Final,neededFeatures2Hyp4Final,neededFeatures3Hyp4Final,neededFeatures4Hyp4Final];
% n = size(X4Final(1,:));
% for w=2:n(2)
%     if max(abs(X4Final(:,w)))~=0
%     X4Final(:,w)=(X4Final(:,w)-mean((X4Final(:,w))))./std(X4Final(:,w));
%     end
% end
% errorFinalHyp4444 = ComputeCost(X4Final,resultsFinal,theta4);





%fprintf('Prediction for house:\t%f\n', ([1, 3 , 3, 2] * theta))

%fprintf('Prediction for house:\t%f\n', ([1, 3 , 3, 2] * normalEquation))



