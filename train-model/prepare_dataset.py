import pandas as pd
import itertools

read_path = "fer2013.csv"
write_path = "output/balanced-fer2013.csv"

def reading(path):
    DataFrame_ = pd.read_csv(path)
    emotion = DataFrame_['emotion']
    pixels = DataFrame_['pixels']
    usage = DataFrame_['Usage']
    return emotion, pixels, usage

def divided(emotion, pixels, usage):
    labels = {
        "angry": {"emotion": [], "pixels": [], "usage": []},
        "disgust": {"emotion": [], "pixels": [], "usage": []},
        "fear": {"emotion": [], "pixels": [], "usage": []},
        "happy": {"emotion": [], "pixels": [], "usage": []},
        "sad": {"emotion": [], "pixels": [], "usage": []},
        "surprise": {"emotion": [], "pixels": [], "usage": []},
        "neutral": {"emotion": [], "pixels": [], "usage": []},
        }

    
    for e, p, u in zip(emotion, pixels, usage):
        if e == 0:
            labels["angry"]["emotion"].append(e)
            labels["angry"]["pixels"].append(p)
            labels["angry"]["usage"].append(u)

        elif e == 1:
            labels["disgust"]["emotion"].append(e)
            labels["disgust"]["pixels"].append(p)
            labels["disgust"]["usage"].append(u)

        elif e == 2:
            e = e - 1
            labels["fear"]["emotion"].append(e)
            labels["fear"]["pixels"].append(p)
            labels["fear"]["usage"].append(u)

        elif e == 3:
            e = e - 1
            labels["happy"]["emotion"].append(e)
            labels["happy"]["pixels"].append(p)
            labels["happy"]["usage"].append(u)

        elif e == 4:
            e = e - 1
            labels["sad"]["emotion"].append(e)
            labels["sad"]["pixels"].append(p)
            labels["sad"]["usage"].append(u)

        elif e == 5:
            e = e - 1
            labels["surprise"]["emotion"].append(e)
            labels["surprise"]["pixels"].append(p)
            labels["surprise"]["usage"].append(u)

        elif e == 6:
            e = e - 1
            labels["neutral"]["emotion"].append(e)
            labels["neutral"]["pixels"].append(p)
            labels["neutral"]["usage"].append(u)

        else:
            print("This is not fer2013 dataset!!")
            exit()

    
    return labels


def prepare(labels):
    
    prepare_labels = {
        "angry": len(labels["angry"]["emotion"]) + (len(labels["happy"]["emotion"]) - len(labels["angry"]["emotion"])),
        "disgust": 0,
        "fear": len(labels["fear"]["emotion"]) + (len(labels["happy"]["emotion"]) - len(labels["fear"]["emotion"])),
        "happy": len(labels["happy"]["emotion"]),
        "sad": len(labels["sad"]["emotion"]) + (len(labels["happy"]["emotion"]) - len(labels["sad"]["emotion"])),
        "surprise": len(labels["surprise"]["emotion"]) + (len(labels["happy"]["emotion"]) - len(labels["surprise"]["emotion"])),
        "neutral": len(labels["neutral"]["emotion"]) + (len(labels["happy"]["emotion"]) - len(labels["neutral"]["emotion"])),
        }
    

    emotions = []
    pixels = []
    usages = []
    counter = 0
    
    for i in range (0,  prepare_labels["angry"]):
        emotions.append(labels["angry"]["emotion"][counter])
        pixels.append(labels["angry"]["pixels"][counter])
        usages.append(labels["angry"]["usage"][counter])
        counter += 1
        if counter == (len(labels["angry"]["emotion"]) - 1):
            counter = 0
            
    
    counter = 0
    for i in range (0,  prepare_labels["disgust"]):
        emotions.append(labels["disgust"]["emotion"][counter])
        pixels.append(labels["disgust"]["pixels"][counter])
        usages.append(labels["disgust"]["usage"][counter])
        counter += 1
        if counter == (len(labels["disgust"]["emotion"]) - 1):
            counter = 0

       
    counter = 0
    for i in range (0,  prepare_labels["fear"]):
        emotions.append(labels["fear"]["emotion"][counter])
        pixels.append(labels["fear"]["pixels"][counter])
        usages.append(labels["fear"]["usage"][counter])
        counter += 1
        if counter == (len(labels["fear"]["emotion"]) - 1):
            counter = 0
            
    
    counter = 0
    for i in range (0,  prepare_labels["happy"]):
        emotions.append(labels["happy"]["emotion"][counter])
        pixels.append(labels["happy"]["pixels"][counter])
        usages.append(labels["happy"]["usage"][counter])
        counter += 1
        if counter == (len(labels["happy"]["emotion"]) - 1):
            counter = 0

   
    counter = 0
    for i in range (0,  prepare_labels["sad"]):
        emotions.append(labels["sad"]["emotion"][counter])
        pixels.append(labels["sad"]["pixels"][counter])
        usages.append(labels["sad"]["usage"][counter])
        counter += 1
        if counter == (len(labels["sad"]["emotion"]) - 1):
            counter = 0

  
    counter = 0
    for i in range (0,  prepare_labels["surprise"]):
        emotions.append(labels["surprise"]["emotion"][counter])
        pixels.append(labels["surprise"]["pixels"][counter])
        usages.append(labels["surprise"]["usage"][counter])
        counter += 1
        if counter == (len(labels["surprise"]["emotion"]) - 1):
            counter = 0
            
    
    counter = 0
    for i in range (0,  prepare_labels["neutral"]):
        emotions.append(labels["neutral"]["emotion"][counter])
        pixels.append(labels["neutral"]["pixels"][counter])
        usages.append(labels["neutral"]["usage"][counter])
        counter += 1
        if counter == (len(labels["neutral"]["emotion"]) - 1):
            counter = 0

            
    dataset = pd.DataFrame(
    {'emotion': emotions,
     'pixels': pixels,
     'Usage': usages,
    })


    return dataset

    



def main():

    print("Dataset Preparing")
    
    print("\tReading ...", end="")
    emotion, pixels, usage = reading(read_path)
    
    print("\n\tDividing ...", end="")
    labels = divided(emotion, pixels, usage)
    
    print("\n\tPreparing ...", end="")
    dataset = prepare(labels)
    
    print("\n\tWriting  ...", end="")
    dataset.to_csv(write_path, index=False)
                   
    print("\nDataset Prepared", end="")
    
main()



