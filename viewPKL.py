import pickle

# Load the synthesizer
with open('models/account_booking/synthesizer.pkl', 'rb') as file:
    synthesizer = pickle.load(file)

# Check its type
print(type(synthesizer))

# View available methods and attributes
print(dir(synthesizer))

# If you want to see the learned metadata:
print(synthesizer.metadata.to_dict())

# If you want to sample synthetic data:
synthetic_data = synthesizer.sample()
print(synthetic_data)