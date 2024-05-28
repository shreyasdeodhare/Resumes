import json

# Original JSON string
json_data = '''
{\"gender\": \"M\", \"address\": \"Solanke Layout, Ward No. 12, Buldana, Tq Buldana, Di Buldana 443001.\", \"email\": \"aakash.s.bhalerao358@gmail.com\", \"skills\": \"\\\"HTML5, CSS, SQL, C, C++, Bootstrap, JavaScript Animation Libraries, MySQL, Microsoft Office suite, Adobe Photoshop (Beginner), Windows (7, 8, 10) user, Android Studio, Visual Studio, VS Code, Sublime Text. Linux (user level).\\\"\", \"city\": \"BULDANA\", \"candidateName\": \"Aakash Bhalerao\", \"dateOfBirth\": \"24-September-1992\", \"phoneNumber\": \"+91-8552939143\", \"altPhoneNumber\": \"+91-8623074807\", \"higherQualification\": \"BACHELOR OF ENGINEERING\", \"passingYear\": \"2020\", \"previousCompanyInformation\": \"\\\"[{'Company_Name': 'Prostorm Innotech Pvt. Ltd., Buldana', 'start_date': 'April 2020', 'end_date': None, 'Duration': '28', 'jobRoleAndResponsibilities': 'Requirements Gathering for Website/Web Application Development.\\\\\\\\n\\\\u2022  Analysis, Planning, Designing, Developing Websites and Web Applications.\\\\\\\\n\\\\u2022  Hosting and Maintenance of Websites and Web Applications.', 'Project_Name': None}, {'Company_Name': 'Prostorm Innotech Pvt. Ltd., Buldana', 'start_date': 'October-2019', 'end_date': 'March-2020', 'Duration': '6', 'jobRoleAndResponsibilities': None, 'Project_Name': None}]\\\"\", \"currentCompanyName\": \"Prostorm Innotech Pvt. Ltd., Buldana\", \"Classification\": \"Web Developer\", \"lastCompany\": \"Prostorm Innotech Pvt. Ltd., Buldana\", \"maximumWorkDuration\": \"28\", \"minimumWorkDuration\": \"6\"}"
'''

# Clean JSON string
cleaned_json_data = json_data.replace('\\"', '"').replace('\\n', '').replace("\\'", "'").replace("\\\\", "\\")

print("Cleaned JSON:", cleaned_json_data)


