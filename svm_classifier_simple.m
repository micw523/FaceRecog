function [test_acc, train_acc, model] = svm_classifier_simple(train_X,train_y,test_X,test_y)  

    model = fitcecoc(train_X, train_y, 'Coding', 'sparserandom'); 
    % train an SVM model with a linear kernel using the training data

    train_pred_y = predict(model, train_X); % use the trained model to classify the training data
    train_acc = sum(train_y == train_pred_y)/length(train_y); % find the classifiction accuracy           
    
    test_pred_y = predict(model, test_X); % use the trained model to classify the testing data
    test_acc = sum(test_y == test_pred_y)/length(test_y); % find the classifiction accuracy  
     
end

