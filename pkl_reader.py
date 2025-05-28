import pickle

with open('data\\vqa_train_10k.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can work with `data`
print(type(
    data))
print(data)