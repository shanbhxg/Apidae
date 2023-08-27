view = pd.read_csv("data/views.csv")
view = view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]
view["pos_com_city"] = view["Position"].map(str) + "  " + view["Company"] +"  "+ view["City"]
view['pos_com_city'] = view['pos_com_city'].str.replace('[^a-zA-Z \n\.]',"")
view['pos_com_city'] = view['pos_com_city'].str.lower()
view = view[['Applicant.ID','pos_com_city']]

exper = pd.read_csv("data/experience.csv")
exper = exper[['Applicant.ID','Position.Name']]
exper['Position.Name'] = exper['Position.Name'].str.replace('[^a-zA-Z \n\.]',"")
exper['Position.Name'] = exper['Position.Name'].str.lower()
exper = exper.sort_values(by=['Applicant.ID'])
exper = exper.fillna(" ")
exper = exper.groupby('Applicant.ID', sort=False)['Position..Name'].apply(' '.join).reset_index()

poi = pd.read_csv("data/positions.csv", sep=',')
poi = poi.sort_values(by='Applicant.ID')
poi = poi.drop('Updated.At', 1)
poi = poi.drop('Created.At', 1)
poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.replace('[^a-zA-z \n\.]',"")
poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.lower()
poi = poi.fillna(" ")
poi = poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()

joint = view.merge(exper_applicant, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
print(joint.shape)
joint = joint.fillna(' ')
joint = joint.sort_values(by='Applicant.ID')

combined = joint.merge(poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
combined = combined.fillna(' ')
combined = combined.sort_values(by='Applicant.ID')
combined["pos_com_city1"] = combined["pos_com_city"].map(str) + combined["Position.Name"] +" "+ combined["Position.Of.Interest"]

final = combined[['Applicant.ID','pos_com_city1']]