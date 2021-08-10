function x = emailFeatures(wordIndices)
% FEATURE VECTOR 
% x = emailFeatures(wordIndices) 
% generats a feature vector for an email, given the word indicies
    vocabLen = 1899;
    x = zeros(vocabLen, 1);
    x(wordIndices) = 1;
end