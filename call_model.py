import pickle

clf = pickle.load(open('model.pkl', 'rb'))

def predict(args):
  x = args["x"]
  y = args["y"]
  z = args["z"]
  
  pred = clf.predict([[x,y,z]])
  
  return pred[0]