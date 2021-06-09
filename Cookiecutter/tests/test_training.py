from main import TrainOREvaluate

train_or_evaluate = TrainOREvaluate()

# test that gradients are changed during training
def test_train_gradients():
    gradients = train_or_evaluate.train()
    print(gradients)
    #for i in (range(len(gradients)-1)):
        #print(gradients[i] != gradients[i+1])
        #assert gradients[i] != gradients[i+1]

test_train_gradients()
