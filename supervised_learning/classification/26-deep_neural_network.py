



def save(self, filename):
    import pickle

    if not isinstance

    if not filename


 @staticmethod
 def load(filename):
    import pickle

    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except FileNotFoundError:
        print("Filename does not exist")
        return