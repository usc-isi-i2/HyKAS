import pickle

def save_dict(f, data):
    with open(f, 'wb') as f:
        pickle.dump(data, f)
