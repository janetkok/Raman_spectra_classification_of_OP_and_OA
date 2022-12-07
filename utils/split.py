from sklearn.model_selection import StratifiedKFold, train_test_split
class StratifiedKFold3(StratifiedKFold):

    def split(self, X, y, groups=None):
        s = super().split(X, y, groups)
        tempy= y
        tempy.reset_index(drop=True,inplace=True)
        for train_indxs, test_indxs in s:
            y_train = tempy[train_indxs]
            train_indxs, val_indxs = train_test_split(train_indxs,stratify=y_train, test_size=0.2,random_state=0)
            yield train_indxs, val_indxs, test_indxs