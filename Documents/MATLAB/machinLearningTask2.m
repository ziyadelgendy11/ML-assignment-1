clear all
close all
clc

data = csvread('heart_DD.csv',1,0);

data60Percent = data(1:151 , :);
dataCV = data(152:202 , :);
dataFinal = data(203:end , :);

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

results60Percent = data60Percent(:,14); %results of data
resultsCV = dataCV(:,1);
resultsCV(:)=(resultsCV(:)-mean((resultsCV(:))))./std(resultsCV(:));
resultsFinal = dataFinal(:,1);
resultsFinal(:)=(resultsFinal(:)-mean((resultsFinal(:))))./std(resultsFinal(:));
length = length(results60Percent);
lengthCV = size(resultsCV);
lengthFinal = size(resultsFinal);
alpha = 0.01;



%Hypothisis one
counter = 1;
for degree=2:2:10
    
%neededFeatures1Hyp1 = data60Percent(:,4:5).^degree;
%neededFeatures2Hyp1 = data60Percent(:,11:12).^degree;
%neededFeatures3Hyp1 = data60Percent(:,18:19).^degree;  
   
X1 = [ones(length, 1),AllFeatures];
n1 = size(X1(1,:));
[iterations1 , Js1 , theta1] = helperMLAss1Logistics(X1,alpha,results60Percent);
thetaVector1(:,counter) = theta1;
normalEquation1(:,counter) = normalEquation(X1, results60Percent);
figure(1)
plot(1: iterations1(2), Js1 , '-b');
title('Hypothesis one');
xlabel('Number of iterations');
ylabel('Error');
Js1(end)

%neededFeatures1Hyp1CV = dataCV(:,4:5).^degree;
%neededFeatures2Hyp1CV = dataCV(:,11:12).^degree;
%neededFeatures3Hyp1CV = dataCV(:,18:19).^degree;

X1CV = [ones(lengthCV(1), 1),AllFeaturesCV.^degree];
n = size(X1CV(1,:));
for w=2:n(2)
    if max(abs(X1CV(:,w)))~=0
    X1CV(:,w)=(X1CV(:,w)-mean((X1CV(:,w))))./std(X1CV(:,w));
    end
end
errorCVHyp1(counter) = computeCostLogistics(X1CV,resultsCV,thetaVector1(:,counter));
counter = counter + 1;
end

optDegree1 = find(errorCVHyp1 == min(errorCVHyp1))*2;

%neededFeatures1Hyp1Final = dataFinal(:,4:5).^optDegree1;
%neededFeatures2Hyp1Final = dataFinal(:,11:12).^optDegree1;
%neededFeatures3Hyp1Final = dataFinal(:,18:19).^optDegree1;

X1Final = [ones(lengthFinal(1), 1),AllFeaturesFinal.^optDegree1(1,1)];

n = size(X1Final(1,:));
for w=2:n(2)
    if max(abs(X1Final(:,w)))~=0
    X1Final(:,w)=(X1Final(:,w)-mean((X1Final(:,w))))./std(X1Final(:,w));
    end
end
errorFinalHyp1 = computeCostLogistics(X1Final,resultsFinal,thetaVector1(:,optDegree1(1,1)/2));





%Hypothisis two
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp2 = data60Percent(:,4:6).^degree;
neededFeatures2Hyp2 = data60Percent(:,8).^degree;
neededFeatures3Hyp2 = data60Percent(:,11:12).^degree;
    
X2 = [ones(length, 1),AllFeatures,neededFeatures1Hyp2,neededFeatures2Hyp2,neededFeatures3Hyp2];
n2 = size(X2(1,:));
[iterations2 , Js2 , theta2] = helperMLAss1Logistics(X2,alpha,results60Percent);
thetaVector2(:,counter) = theta2;
normalEquation2(:,counter) = normalEquation(X2, results60Percent);
figure(2)
plot(1: iterations2(2), Js2 , '-b');
title('Hypothesis two');
xlabel('Number of iterations');
ylabel('Error');
Js2(end)

neededFeatures1Hyp2CV = dataCV(:,4:6).^degree;
neededFeatures2Hyp2CV = dataCV(:,8).^degree;
neededFeatures3Hyp2CV = dataCV(:,11:12).^degree;
X2CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp2CV,neededFeatures2Hyp2CV,neededFeatures3Hyp2CV];
n = size(X2CV(1,:));
for w=2:n(2)
    if max(abs(X2CV(:,w)))~=0
    X2CV(:,w)=(X2CV(:,w)-mean((X2CV(:,w))))./std(X2CV(:,w));
    end
end
errorCVHyp2(counter) = computeCostLogistics(X2CV,resultsCV,thetaVector2(:,counter));
counter = counter + 1;
end


optDegree2 = find(errorCVHyp2 == min(errorCVHyp2))*2;

neededFeatures1Hyp2Final = dataFinal(:,4:6).^optDegree2(1,1);
neededFeatures2Hyp2Final = dataFinal(:,8).^optDegree2(1,1);
neededFeatures3Hyp2Final = dataFinal(:,10:11).^optDegree2(1,1);

X2Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp2Final,neededFeatures2Hyp2Final,neededFeatures3Hyp2Final];

n = size(X2Final(1,:));
for w=2:n(2)
    if max(abs(X2Final(:,w)))~=0
    X2Final(:,w)=(X2Final(:,w)-mean((X2Final(:,w))))./std(X2Final(:,w));
    end
end
errorFinalHyp2 = computeCostLogistics(X2Final,resultsFinal,thetaVector2(:,optDegree2(1,1)/2));






%Hypothisis three
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp3 = data60Percent(:,1:3).^degree;
    
X3 = [ones(length, 1),AllFeatures,neededFeatures1Hyp3];
n3 = size(X1(1,:));
[iterations3 , Js3 , theta3] = helperMLAss1Logistics(X3,alpha,results60Percent);
thetaVector3(:,counter) = theta3;
normalEquation3(:,counter) = normalEquation(X3, results60Percent);
figure(3)
plot(1: iterations3(2), Js3 , '-b');
title('Hypothesis three');
xlabel('Number of iterations');
ylabel('Error');
Js3(end)

neededFeatures1Hyp3CV = dataCV(:,1:3).^degree;
X3CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp3CV];
n = size(X3CV(1,:));
for w=2:n(2)
    if max(abs(X3CV(:,w)))~=0
    X3CV(:,w)=(X3CV(:,w)-mean((X3CV(:,w))))./std(X3CV(:,w));
    end
end
errorCVHyp3(counter) = computeCostLogistics(X3CV,resultsCV,thetaVector3(:,counter));
counter = counter + 1;
end

optDegree3 = find(errorCVHyp3 == min(errorCVHyp3))*2;

neededFeatures1Hyp3Final = dataFinal(:,1:3).^optDegree3(1,1);


X3Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp3Final];

n = size(X3Final(1,:));
for w=2:n(2)
    if max(abs(X3Final(:,w)))~=0
    X3Final(:,w)=(X3Final(:,w)-mean((X3Final(:,w))))./std(X3Final(:,w));
    end
end
errorFinalHyp3 = computeCostLogistics(X3Final,resultsFinal,thetaVector3(:,optDegree3(1,1)/2));






%Hypothisis four
counter = 1;
for degree=2:2:10
    
neededFeatures1Hyp4 = data60Percent(:,6).^degree;
neededFeatures2Hyp4 = data60Percent(:,9:10).^degree;
neededFeatures3Hyp4 = data60Percent(:,2).^degree;
neededFeatures4Hyp4 = data60Percent(:,12).^degree;
    
X4 = [ones(length, 1),AllFeatures,neededFeatures1Hyp4];
n4 = size(X1(1,:));
[iterations4 , Js4 , theta4] = helperMLAss1Logistics(X4,alpha,results60Percent);
thetaVector4(:,counter) = theta4;
normalEquation4(:,counter) = normalEquation(X4, results60Percent);
figure(4)
plot(1: iterations4(2), Js4 , '-b');
title('Hypothesis four');
xlabel('Number of iterations');
ylabel('Error');
Js4(end)

neededFeatures1Hyp4CV = dataCV(:,6).^degree;
neededFeatures2Hyp4CV = dataCV(:,9:10).^degree;
neededFeatures3Hyp4CV = dataCV(:,2).^degree;
neededFeatures4Hyp4CV = dataCV(:,12).^degree;
X4CV = [ones(lengthCV(1), 1),AllFeaturesCV,neededFeatures1Hyp4CV];
n = size(X4CV(1,:));
for w=2:n(2)
    if max(abs(X4CV(:,w)))~=0
    X4CV(:,w)=(X4CV(:,w)-mean((X4CV(:,w))))./std(X4CV(:,w));
    end
end
errorCVHyp4(counter) = computeCostLogistics(X4CV,resultsCV,thetaVector4(:,counter));
counter = counter + 1;
end

optDegree4 = find(errorCVHyp4 == min(errorCVHyp4))*2;

neededFeatures1Hyp4Final = dataFinal(:,6).^optDegree4(1,1);
neededFeatures2Hyp4Final = dataFinal(:,9:10).^optDegree4(1,1);
neededFeatures3Hyp4Final = dataFinal(:,2).^optDegree4(1,1);
neededFeatures4Hyp4Final = dataFinal(:,12).^optDegree4(1,1);


X4Final = [ones(lengthFinal(1), 1),AllFeaturesFinal,neededFeatures1Hyp4Final];

n = size(X4Final(1,:));
for w=2:n(2)
    if max(abs(X4Final(:,w)))~=0
    X4Final(:,w)=(X4Final(:,w)-mean((X4Final(:,w))))./std(X4Final(:,w));
    end
end
errorFinalHyp4 = computeCostLogistics(X4Final,resultsFinal,thetaVector4(:,optDegree4(1,1)/2));

