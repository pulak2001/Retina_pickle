from flask import * 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os, pickle
app = Flask(__name__, static_folder='images') 
app.secret_key='asdsdfsdfs13sdf_df%&'
# model = tf.keras.models.load_model('model/CNN1.h5', custom_objects={'KerasLayer': hub.KerasLayer})
# model1 = tf.keras.models.load_model('model/effnet_b5_model.h5')
users = {
    "pulakkumarghosh2001@gmail.com": "Pulak@2001",
    "admin@infomaticae.com": "admin",
}

@app.route('/login', methods=['POST', 'GET'])
def login():
    if "username" in session:
        return redirect(url_for('main'))
    if request.method == 'GET':
        return render_template('login.html')
    if request.method == 'POST':
        username = request.form.get('email')
        password = request.form.get('password')
        print(username)
        print(users)
        print(username in users)
        if username in users:
            pass_org = users[username]
            print(pass_org)
            if pass_org == password:
                session['username'] = username
                return redirect(url_for('main'))
            else:
                return render_template('login.html', msg="Wrong Credentials !")
        else:
            return render_template('login.html', msg="Wrong Credentials !")

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')  
def main():  
    print(session)
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", modelType=0)  
  
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        folder = 'images'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        dicts = {
        "DR" : "Diabetic retinopathy",
        "ARMD" : "Age-related macular degeneration",
        "MH" : "Media haze",
        "DN" : "Drusen",
        "MYA" : "Myopia",
        "BRVO" : "Branch retinal vein occlusion",
        "TSLN" : "Tessellation",
        "ERM" : "Epiretinal membrane",
        "LS" : "Laser scars",
        "MS" : "Macular scars",
        "CSR" : "Central serous retinopathy",
        "ODC" :  "Optic disc cupping",
        "CRVO" : "Central retinal vein occlusion",
        "TV" : "Tortuous vessels",
        "AH" : "Asteroid hyalosis",
        "ODP" : "Optic disc pallor",
        "ODE" : "Optic disc edema",
        "ST" : "Optociliary shunt",
        "AION" : "Anterior ischemic optic neuropathy",
        "PT" : "Parafoveal telangiectasia",
        "RT" : "Retinal traction",
        "RS" : "Retinitis",
        "CRS" : "Chorioretinitis",
        "EDN" : "Exudation",
        "RPEC" : "Retinal pigment epithelium changes",
        "MHL" : "Macular hole",
        "RP" : "Retinitis pigmentosa",
        "CWS" : "Cotton-wool spots",
        "CB" : "Coloboma",
        "ODPM" : "Optic disc pit maculopathy",
        "PRH" : "Preretinal hemorrhage",
        "MNF" : "Myelinated nerve fibers",
        "HR" : "Hemorrhagic retinopathy",
        "CRAO" : "Central retinal artery occlusion",
        "TD" : "Tilted disc",
        "CME" : "Cystoid macular edema",
        "PTCR" : "Post-traumatic choroidal rupture",
        "CF" : "Choroidal folds",
        "VH" : "Vitreous hemorrhage",
        "MCA" : "Macroaneurysm",
        "VS" : "Vasculitis",
        "BRAO" : "Branch retinal artery occlusion",
        "PLQ" :"Plaque",
        "HPED" : "Hemorrhagic pigment epithelial detachment",
        "CL" : "Collateral"
        }
        classes = ['DR','ARMD','MH','DN','MYA','BRVO','TSLN','ERM','LS','MS','CSR','ODC','CRVO','TV','AH','ODP','ODE','ST','AION','PT','RT','RS','CRS','EDN','RPEC','MHL','RP','CWS','CB','ODPM','PRH','MNF','HR','CRAO','TD','CME','PTCR','CF','VH','MCA','VS','BRAO','PLQ','HPED','CL']
        classes = [dicts[i] for i in classes]
        f = request.files['file']
        model_choose = request.form.get('model')
        try:
            model_choose = int(model_choose)
        except:
            return render_template('index.html', modelType=1)
        f.save('images/'+f.filename)
        img_path = 'images/'+f.filename
        if str(model_choose) == '1':
            with open('pickle/effnet_b5_model.pkl', 'rb') as file:
                model_architecture, model_weights = pickle.load(file)
                loaded_model = tf.keras.models.model_from_json(model_architecture)
                loaded_model.set_weights(model_weights)
                img1 = tf.keras.preprocessing.image.load_img(img_path, target_size=(456, 456))
                img1 = tf.keras.preprocessing.image.img_to_array(img1)
                img1 = np.expand_dims(img1, axis=0)
                img1 = img1/255.
                category = 0
                proba2 = loaded_model.predict(img1)
                category = proba2[0][0]
                print(category)
                category_score = category
                category = round(category)
                category_val = category
                class2 = {
                    0: 'None',
                    1: 'Mild',
                    2: 'Moderate',
                    3: 'Severe',
                    4: 'Proliferative'
                }
                severity = class2[category]
                return render_template('result2.html', name=f.filename, link=f.filename, disease={}, severity=severity, severity_val=category_val, model=model_choose, severite_score=category_score)
        else:
            with open('pickle/CNN.pkl', 'rb') as file:
                model_architecture, model_weights = pickle.load(file)
                loaded_model = tf.keras.models.model_from_json(model_architecture,  custom_objects={'KerasLayer': hub.KerasLayer})

                # Set the model weights
                loaded_model.set_weights(model_weights)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 3))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img/255.
                img = np.expand_dims(img, axis=0)
                proba = loaded_model.predict(img)
                sorted_categories = np.argsort(proba[0])
                disease = {}
                classes = np.array(classes)
                for i in range(len(classes)):
                    if (round(proba[0][sorted_categories[i]], 3)) >= 0.1:
                        disease[classes[sorted_categories[i]]] = (round(proba[0][sorted_categories[i]]*100, 2))
                print(disease)
                return render_template("result2.html", name = f.filename, link=f.filename, disease=disease, severity='severity', severity_val=0, model=model_choose, severite_score=0)  
    
if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=5000, debug=True)