from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def build_ensemble():
    
    logreg = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,
        random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)

    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('knn', knn),
            ('logreg', logreg)
        ],
        voting='soft',
        n_jobs=-1
    )

    return ensemble
