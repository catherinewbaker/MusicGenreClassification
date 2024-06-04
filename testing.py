from knn import knn, knn_
from nb import nb, nb_
from sgd import sgd, sgd_

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

from itertools import combinations

from initialPreprocessing import gen_Train_and_Test, top_tracks, top_n_genre_tracks, top_echonest_tracks, top_tracks_final

from track_name import process_track_names, vectorise


def runTests(sample, featSelection,processedX=None, features=[],K=-1):

    if processedX is not None:
        if len(features)>0:
            X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0, processedX,features)
            print(f"\n\nTesting for Track Name and {features}")
            resultString = f"Track Name,{features}"
        else:
            X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0, processedX)
            print(f"\n\nTesting for Track Name")
            resultString = f"Track Name,"
    elif len(features)>0:
        X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,features)
        print(f"\n\nTesting for {features}")
        resultString = f"{features},"
    else:
        X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,featSelection,0)
        print(f"\n\nTesting for {featSelection}")
        resultString = f"{featSelection},"

    
    nb_acc, nb_pred = nb(X_train, X_test, y_train, y_test)
    sgd_acc, sgd_pred = sgd(X_train, X_test, y_train, y_test)
    if K<0:
        knn_acc, knn_pred, knn_bestk = knn(X_train, X_test, y_train, y_test)
    else:
        # knn_acc, knn_pred, knn_bestk = knn(X_train, X_test, y_train, y_test,"",K)
        knn_acc = 0.00
        knn_bestk = "NA"

    resultString += f"{nb_acc:.2f},{sgd_acc:.2f},{knn_acc:.2f},{knn_bestk}\n"
    print(resultString)
    return resultString

def Boost(sample, featSelection,processedX=None, features=[],K=1,hard=True):

    resultString = ''
    if processedX is not None:
        if len(features)>0:
            X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0, processedX,features)
            print(f"\n\nBuilding for Track Name and {features}")
            resultString = f"Track Name,{features}"
        else:
            X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0, processedX)
            print(f"\n\nBuilding for Track Name")
            resultString = f"Track Name,"
    elif len(features)>0:
        X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,features)
        print(f"\n\nBuilding for {features}")
        # resultString = f"{features},"
    else:
        X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,featSelection,0)
        print(f"\n\nBuilding for {featSelection}")
        resultString = f"{featSelection},"

    
    nb_clf, nb_pred = nb_(X_train, X_test, y_train, y_test)
    sgd_clf, sgd_pred = sgd_(X_train, X_test, y_train, y_test,hard)
    knn_clf, knn_pred = knn_(X_train, X_test, y_train, y_test,K)

    voting_type = 'soft'
    if hard == True:
        voting_type = 'hard'

    voting_clf = VotingClassifier(
        estimators=[('knn', knn_clf), ('sgd', sgd_clf), ('nb', nb_clf)],
        voting=voting_type
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) # Generate report of the values
    
    # Print the results
    print("Ensemble")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    resultString+=f"{accuracy:.2f}\n"
    return resultString



def runSingleFeatures(datasetindex=0):
    tracks = top_tracks_final()

    fullResults = ""

    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 1 or i == 3:
            single_feats = ['track_duration','track_listens','track_favorites','days_since_first']
        else:
            single_feats = ['track_duration','track_listens','track_favorites']
    

        for feat in single_feats:
            fullResults+=runTests(tracks[i],feat)
        
        fullResults+="\n"


    
    print(fullResults)

def boostSingleFeatures(datasetindex=0):
    tracks = top_tracks_final()

    fullResults = ""

    bestks = [[153,107,115],[41,35,15,3],[77,73,65],[37,33,31,1]]
    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 1 or i == 3:
            single_feats = ['track_duration','track_listens','track_favorites','days_since_first']
        else:
            single_feats = ['track_duration','track_listens','track_favorites']
    
        j = 0
        for feat in single_feats:
            fullResults+=Boost(tracks[i],feat,None,[],bestks[i][j])
            j+=1
        
        fullResults+="\n"


    
    print(fullResults)

def runSingleEchoFeatures(datasetindex=0):
    tracks = top_tracks_final()

    single_features = ['echonest_acousticness','echonest_danceability','echonest_energy','echonest_instrumentalness','echonest_liveness','echonest_speechiness','echonest_tempo','echonest_valence']

    fullResults = "\n DATASET 2 \n"

    for feat in single_features:
        fullResults+=runTests(tracks[2],feat)
        
    fullResults+="\n DATASET 3 \n"

    for feat in single_features:
        fullResults+=runTests(tracks[3],feat)
        
    fullResults+="\n"
    
    print(fullResults)

def runTrackName():
    tracks = top_tracks_final()

    fullResults = ""

    for i in range(4):
        fullResults += "\n DATASET "+str(i)+"\n"
        sample = process_track_names(tracks[i], True, False, False)
        vec_sample = vectorise(sample,'tfidf')
        fullResults += runTests(tracks[i],"",vec_sample)

        fullResults += "\n"

    print(fullResults)

def generate_feature_combos(features):
    combos = []
    for r in range(2, len(features) + 1):
        combos.extend(combinations(features, r))
    return combos

def runFeatureCombos():
    tracks = top_tracks_final()

    fullResults = ""

    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 1 or i == 3:
            single_feats = ['track_duration','track_listens','track_favorites','days_since_first']
        else:
            single_feats = ['track_duration','track_listens','track_favorites']
    
        feat_combos = generate_feature_combos(single_feats)

        for feat in feat_combos:
            fullResults+=runTests(tracks[i],'',None,np.array(feat))
        
        fullResults+="\n"


    
    print(fullResults)

def boostSimpleFeatures(hard=True):
    tracks = top_tracks_final()

    fullResults = ""

    bestks = [[79,131,145,79],[45,41,31,13,1,1,35,33,37,7,29],[69,59,65,73],[31,29,19,19,1,1,29,13,17,9,11]]
    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 1 or i == 3:
            single_feats = ['track_duration','track_listens','track_favorites','days_since_first']
        else:
            single_feats = ['track_duration','track_listens','track_favorites']
    
        feat_combos = generate_feature_combos(single_feats)

        j = 0
        for feat in feat_combos:
            fullResults+=Boost(tracks[i],'',None,np.array(feat),bestks[i][j],hard)
        
        fullResults+="\n"


    
    print(fullResults)

def runEchoFeatureCombos():
    tracks = top_tracks_final()

    fullResults = ""

    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 2 or i == 3:
            single_feats = ['echonest_acousticness','echonest_danceability','echonest_energy','echonest_instrumentalness','echonest_liveness','echonest_speechiness','echonest_tempo','echonest_valence']
        else:
            continue
    
        feat_combos = generate_feature_combos(single_feats)

        for feat in feat_combos:
            fullResults+=runTests(tracks[i],'',None,np.array(feat))
        
        fullResults+="\n"


    
    print(fullResults)

def boostEchoFeatures(hard=True):
    tracks = top_tracks_final()

    fullResults = ""
    bestks = [[],[],[77, 41, 41, 39, 47, 29, 23, 43, 39, 21, 39, 19, 17, 17, 29, 21, 29, 13, 37, 15, 23],[21, 17, 17, 19, 15, 17, 11, 23, 9, 29, 11, 15, 9, 13, 9, 13, 13, 19, 11, 9, 11]]
       
    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 2:
            feat_combos = [
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_liveness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence']
                ]
        elif i==3:
            feat_combos = [
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_liveness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence']
            ]

        else:
            continue

        
        j=0
        for feat in feat_combos:
            fullResults+=Boost(tracks[i],'',None,np.array(feat),bestks[i][j],hard)
            j+=1
        
        fullResults+="\n"

    
    print(fullResults)

def runEchoPlusSimpleFeatureCombos():
    tracks = top_tracks_final()

    fullResults = ""

    with open('results.txt','w') as file:
        for i in range(4):
            fullResults += "DATASET "+str(i) +"\n"
            file.write("DATASET "+str(i) +"\n")
            echo_feats = ['echonest_acousticness','echonest_danceability','echonest_energy','echonest_instrumentalness','echonest_liveness','echonest_speechiness','echonest_tempo','echonest_valence']

            if i == 3:
                single_feats = ['track_duration','track_listens','track_favorites','days_since_first','echonest_acousticness','echonest_danceability','echonest_energy','echonest_instrumentalness','echonest_liveness','echonest_speechiness','echonest_tempo','echonest_valence']
                orig_feats = ['track_duration','track_listens','track_favorites','days_since_first']
            elif i==2:
                single_feats = ['track_duration','track_listens','track_favorites','echonest_acousticness','echonest_danceability','echonest_energy','echonest_instrumentalness','echonest_liveness','echonest_speechiness','echonest_tempo','echonest_valence']
                orig_feats = ['track_duration','track_listens','track_favorites']
            else:
                continue

            simple_feat_combos = generate_feature_combos(orig_feats)
            echo_feat_combos = generate_feature_combos(echo_feats)
            feat_combos = generate_feature_combos(single_feats)

            # print(f"simple feat combos {len(simple_feat_combos)} echo feat combos {len(echo_feat_combos)} all feat combos {len(feat_combos)}")

            # Convert the combinations to sets for easier comparison
            simple_feat_set = set(map(tuple, simple_feat_combos))
            echo_feat_set = set(map(tuple, echo_feat_combos))

            # Filter feat_combos to remove combinations that appear in simple_feat_set or echo_feat_set
            filtered_feat_combos = [combo for combo in feat_combos if tuple(combo) not in simple_feat_set and tuple(combo) not in echo_feat_set]

            # print(f"Filtered {len(filtered_feat_combos)}")
            cnt = 0
            for feat in filtered_feat_combos:
                print(cnt)
                cnt+=1
                result = runTests(tracks[i],'',None,np.array(feat))
                fullResults+= result
                file.write(result)
            
            file.write("\n")
            fullResults+="\n"


    
    # print(fullResults)

def runSimpleAndTrackName():
    tracks = top_tracks_final()

    fullResults = ""

    for i in range(4):
        fullResults += "\n DATASET "+str(i)+"\n"
        sample = process_track_names(tracks[i], True, False, False)
        vec_sample = vectorise(sample,'tfidf')
        
        if i == 1 or i == 3:
            single_feats = ['track_duration','track_listens','track_favorites','days_since_first']
        else:
            single_feats = ['track_duration','track_listens','track_favorites']
    
        feat_combos = generate_feature_combos(single_feats)

        for feat in feat_combos:
            fullResults+=runTests(tracks[i],'',vec_sample,np.array(feat),1)
        
        fullResults+="\n"
        
        # fullResults += runTests(tracks[i],"",vec_sample)

        # fullResults += "\n"

    print(fullResults)

def boostOptimalFeatures(hard=True):
    tracks = top_tracks_final()

    fullResults = ""
    bestks = [[],[],[17, 17, 27, 17, 15, 27, 27, 27, 15],[37, 25, 33, 23, 15, 13, 13, 7, 15, 13, 21, 11, 13, 11, 13, 15, 13, 7, 13, 11, 5]]
       
    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 2:
            feat_combos = [
                ['track_duration', 'echonest_acousticness', 'echonest_danceability', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence'],
                ['track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_valence'],
                ['track_duration', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo'],
                ['track_listens', 'track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_valence'],
                ['track_listens', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_valence'],
                ['track_duration', 'track_listens', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo'],
                ['track_duration', 'track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo'],
                ['track_duration', 'track_listens', 'track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo'],
                ['track_duration', 'track_listens', 'track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_valence']
            ]
        elif i==3:
            feat_combos = [
                ['days_since_first', 'echonest_acousticness', 'echonest_energy'],
                ['days_since_first', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy'],
                ['days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness'],
                ['track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness'],
                ['track_favorites', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_listens', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness'],
                ['track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness'],
                ['track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['track_duration', 'track_listens', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_duration', 'track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_liveness', 'echonest_speechiness'],
                ['track_duration', 'track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_listens', 'track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_listens', 'days_since_first', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_liveness', 'echonest_speechiness'],
                ['track_listens', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo'],
                ['track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_speechiness', 'echonest_tempo'],
                ['track_favorites', 'echonest_acousticness', 'echonest_danceability', 'echonest_energy', 'echonest_instrumentalness', 'echonest_speechiness', 'echonest_tempo'],
                ['track_duration', 'track_listens', 'track_favorites', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_speechiness', 'echonest_valence'],
                ['track_duration', 'track_listens', 'days_since_first', 'echonest_acousticness', 'echonest_energy', 'echonest_instrumentalness', 'echonest_liveness', 'echonest_speechiness', 'echonest_tempo', 'echonest_valence']
            ]

        else:
            continue

        
        j=0
        for feat in feat_combos:
            fullResults+=Boost(tracks[i],'',None,np.array(feat),bestks[i][j],hard)
            j+=1
        
        fullResults+="\n"

    
    print(fullResults)

def boost():
    print("TRYING TO BOOST")

    
# runSimpleAndTrackName()

# runSingleFeatures()
# runSingleEchoFeatures()
# runTrackName()

# boostSimpleFeatures(True)
# boostEchoFeatures(True)
boostOptimalFeatures(True)


# runEchoPlusSimpleFeatureCombos()




