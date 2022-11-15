import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_titanic(survived, pclass_num_max, pclass_num_min, sex_female, sex_male, 
                    age_max, age_min, sibsp_max, sibsp_min,parch_max,parch_min,embarked_max,
                     embarked_min,fare_per_customer_max,fare_per_customer_min,cabin_have,cabin_none):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [random.random_integers(pclass_num_max, pclass_num_min)],
                       "sex": [random.random_integers(sex_female, sex_male)],
                       "age": [random.uniform(age_max, age_min)],
                       "sibsp": [random.random_integers(sibsp_max, sibsp_min)],
                       "parch":[random.random_integers(parch_max,parch_min)],
                       "embarked":[random.random_integers(embarked_max,embarked_min)],
                       "fare_per_customer":[random.uniform(fare_per_customer_max,fare_per_customer_min)],
                       "cabin":[random.random_integers(cabin_have,cabin_none)]
                      })
    df['survived'] = survived
    return df


def get_random_titanic():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    survived_df = generate_titanic("S", 3, 1, 1, 0, 30, 0, 4, 0,5,0,3,1,200,0,1,0)
    deceased_df = generate_titanic("D", 3, 1, 1, 0, 80, 50, 4, 0,5,0,3,1,200,0,1,0)

    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        titanic_df = survived_df
        print("Survived added")
    else:
        titanic_df = deceased_df
        print("Deceased added")

    return titanic_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_titanic()

    titanic_fg = fs.get_feature_group(name="titanic_survival_modal",version=1)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
