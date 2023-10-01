import joblib
import torch
import torch.nn as nn
import fire


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.layers(x)


class Prediction:
    def __init__(self):
        self.scaler = joblib.load("../model/scaler.pkl")
        self.model = BinaryClassifier(13)
        self.model.load_state_dict(torch.load("../model/nn_model.pt"))
        self.model.eval()

    def predict(self, x):
        x_transform = self.scaler.transform(x)
        x_tensor = torch.FloatTensor(x_transform)
        y_pred = self.model(x_tensor)
        with torch.no_grad():
            if y_pred > 0.5:
                return 1
            else:
                return 0


class command_interface:
    def __init__(
        self,
        age=0,
        sex=0,
        cp=0,
        trestbps=0,
        chol=0,
        fbs=0,
        restecg=0,
        thalach=0,
        exang=0,
        oldpeak=0,
        slope=0,
        ca=0,
        thal=0,
    ):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def __get_data(self):
        input = [
            [
                self.age,
                self.sex,
                self.cp,
                self.trestbps,
                self.chol,
                self.fbs,
                self.restecg,
                self.thalach,
                self.exang,
                self.oldpeak,
                self.slope,
                self.ca,
                self.thal,
            ]
        ]
        return input

    def __range_float(self, value, min, max):
        if value >= min and value <= max:
            return True
        else:
            return False

    def __check_data(self):
        if (
            self.age
            == 0 & self.sex
            == 0 & self.cp
            == 0 & self.trestbps
            == 0 & self.chol
            == 0 & self.fbs
            == 0 & self.restecg
            == 0 & self.thalach
            == 0 & self.exang
            == 0 & self.oldpeak
            == 0 & self.slope
            == 0 & self.ca
            == 0 & self.thal
            == 0
        ):
            assert False, "Please input the data"
        elif self.age not in range(0, 101):
            assert False, "Wrong age, check the help"
        elif self.sex not in range(0, 2):
            assert False, "Wrong sex, check the help"
        elif self.cp not in range(0, 4):
            assert False, "Wrong cp, check the help"
        elif self.trestbps not in range(90, 201):
            assert False, "Wrong trestbps, check the help"
        elif self.chol not in range(100, 601):
            assert False, "Wrong chol, check the help"
        elif self.fbs not in range(0, 2):
            assert False, "Wrong fbs, check the help"
        elif self.restecg not in range(0, 3):
            assert False, "Wrong restecg, check the help"
        elif self.thalach not in range(70, 221):
            assert False, "Wrong thalach, check the help"
        elif self.exang not in range(0, 2):
            assert False, "Wrong exang, check the help"
        elif self.__range_float(self.oldpeak, 0.0, 6.3) == False:
            assert False, "Wrong oldpeak, check the help"
        elif self.slope not in range(0, 3):
            assert False, "Wrong slope, check the help"
        elif self.ca not in range(0, 4):
            assert False, "Wrong ca, check the help"
        elif self.thal not in range(0, 4):
            assert False, "Wrong thal, check the help"
        else:
            return True

    def pred(self):
        self.__check_data()
        input = self.__get_data()
        pred = Prediction()
        result = pred.predict(input)
        result = (
            "You have a heart disease"
            if result == 1
            else "You don't have a heart disease"
        )
        return result

    # 帮助文档
    def help(self):
        doc = """
        heart disease prediction
         ██╗    ██████╗    ██████╗    ██╗  ██╗  
        ███║   ██╔═████╗   ╚════██╗   ██║  ██║  
        ╚██║   ██║██╔██║    █████╔╝   ███████║  
         ██║   ████╔╝██║   ██╔═══╝    ╚════██║  
         ██║   ╚██████╔╝   ███████╗        ██║  
         ╚═╝    ╚═════╝    ╚══════╝        ╚═╝  
                  __      __        __    /     
            |\ | /  \    |__) |  | / _`  /      
            | \| \__/    |__) \__/ \__> .       
                                               
         Usage:
           python prediction.py help
           python prediction.py pred --age=<age> --sex=<sex> --cp=<cp> --trestbps=<trestbps> --chol=<chol> --fbs=<chol> --restecg=<restecg> --thalach=<thalach> --exang=<exang> --oldpeak=<oldpeak> --slope=<slope> --ca=<ca> --thal=<thal>
         Options:
           --age                  age from 0 to 100
           --sex                  0 is female, 1 is male 
           --cp                   chest pain type, there are four types of chest pain: 0, 1, 2, 3
           --trestbps             resting blood pressure (in mm Hg on admission to the hospital), from 90 to 200
           --chol                 serum cholestoral in mg/dl, from 100 to 600
           --fbs                  (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
           --restecg              resting electrocardiographic results, there are three types of results: 0, 1, 2
           --thalach              maximum heart rate achieved, from 70 to 220
           --exang                exercise induced angina (1 = yes; 0 = no)
           --oldpeak              ST depression induced by exercise relative to rest, from 0.0 to 6.2
           --slope                the slope of the peak exercise ST segment, there are three types of slope: 0, 1, 2
           --ca                   number of major vessels (0-3) colored by flourosopy, from 0 to 3
           --thal                 thalassemia, there are three types of thalassemia: 0, 1, 2, 3
        """
        return doc


if __name__ == "__main__":
    fire.Fire(command_interface)
