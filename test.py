def test_model(model, test_data):
    X_test = model.predict(test_data[0])
    y_test = test_data[1]

    count = 0
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_test[i]))
        if (X_test[i] == y_test[i]):
            count += 1

    print("Accuracy: ", count/len(X_test)*100)
