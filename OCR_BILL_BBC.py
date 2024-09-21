
image=cv2.imread('bill3.png',0)

#convert it into text
text=(pytesseract.image_to_string(image)).lower()
print(text)
#identify the date

match=re.findall(r'\d+[/.-]\d+[/.-]\d+', text)

st=" "
st=st.join(match)
print(st)
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)
sent_tokens=nltk.sent_tokenize(text)
sent_tokens[0].splitlines()[0]

price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
price = list(map(float,price)) 
print(max(price))
x=max(price)  
#we will remove punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(text)
print(new_words)
#stop_words = set(nltk.corpus.stopwords.words('english')) 
nltk.download('stopwords')
#so we need to filter them
stop_words = set(nltk.corpus.stopwords.words('english')) 
#there is the filetred list
filtered_list=[w for w in new_words if w not in stop_words ]
print(filtered_list)
in the next six blocks we will make list of 6 categories.these list will contain few words which are relatable to the category of bill.for instance suppose the above bill of the restuarant would include words like kitchen,food or restuarant.so we will compare it list category and likewise allot the rescpective category.
#entertainment
entertainment = [] 
for syn in wordnet.synsets("entertainment"): 
    for l in syn.lemmas(): 
        entertainment.append(l.name()) 
        
l=['happy','restaurant','food','kitchen','hotel','room','park','movie','cinema','popcorn','combo meal','coffee','mall']
entertainment=entertainment+l

#home utility
home_utility=[] 
for syn in wordnet.synsets("home"): 
    for l in syn.lemmas(): 
         home_utility.append(l.name()) 
l2=['internet','telephone','elecricity','meter','wifi','broadband','consumer','reading','gas','water','postpaid','prepaid']
home_utility+=l2
#grocery
 
grocery=[] 
for syn in wordnet.synsets("grocery"): 
    for l in syn.lemmas(): 
         grocery.append(l.name())
l3=['bigbasket','milk','atta','sugar','suflower','oil','bread','vegetabe','fruit','salt','paneer']
grocery+=l3

#investment
investment=[] 
for syn in wordnet.synsets("investment"): 
    for l in syn.lemmas(): 
         investment.append(l.name()) 
l1=['endowment','grant','loan','applicant','income','expenditure','profit','interest','expense','finance','property','money','fixed','deposit','kissan','vikas']
investment=investment+l1
#travel and transportation
transport=[]
for syn in wordnet.synsets("car"): 
    for l in syn.lemmas(): 
         transport.append(l.name()) 
l4=['cab','ola','uber','autorickshaw','railway','air','emirates','aerofloat','taxi','booking','road','highway']
transport+=l4
#shopping
shopping=[]
for syn in wordnet.synsets("dress"): 
    for l in syn.lemmas(): 
         shopping.append(l.name()) 
l4=['iphone','laptop','saree','max','pantaloons','westside','vedic','makeup','lipstick','cosmetics','mac','facewash','heels','crocs','footwear','purse']
shopping+=l4
#here we will check that the bill belongs to which category
#we will make that category true.
for word in filtered_list:
    if word in entertainment:
        e=True
        break
    elif word in investment:
        inv=True
        break
    elif word in grocery:
        g=True
        break
    elif word in shopping:
        s=True
        break
    elif word in transport:
        t=True
        break
    elif word in home_utility:
        h=True
        break
            
#this is how i created all the csv files.
'''with open('entertainment1.csv', 'a', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['date','organisation','amount'])'''
   
   
#formatting
if(e):
    print("entertainment category")
    filename='{}.csv'.format('entertainment')
    #df=pd.read_csv('entertainment.csv')
elif(inv):
    print("investment category")
    filename='{}.csv'.format('investment')
    #df=pd.read_csv('investment.csv')
elif(s):
    print("shopping category")
    filename='{}.csv'.format('shopping')
    #df=pd.read_csv('shopping.csv')
elif(g):
    print("grocery category")
    filename='{}.csv'.format('grocery')
    #df=pd.read_csv('grocery.csv')
elif(t):
    print("transport category")
    filename='{}.csv'.format('transport')
    #df=pd.read_csv('transport.csv')
elif(h):
    print("home utility category")
    filename='{}.csv'.format('home')
    #df=pd.read_csv('home.csv')
else:
    print("others")
    filename='{}.csv'.format('others')
    #df=pd.read_csv('others.csv')
        
        

#add the contents in thier respective csv file
row_contents = [st,head,x]
from csv import writer
 
def append_list_as_row(file, list_of_elem):
   
    with open(file, 'a+', newline='') as write_obj:
       
        csv_writer = writer(write_obj)
        
        csv_writer.writerow(list_of_elem)
append_list_as_row(filename, row_contents)
#after this make sure you save it
entertainment=pd.read_csv('entertainment.csv')
investment=pd.read_csv('investment.csv')
shopping=pd.read_csv('shopping.csv')
grocery=pd.read_csv('grocery.csv')
transport=pd.read_csv('transport.csv')
other=pd.read_csv('others.csv')
home=pd.read_csv('home.csv')
entertainment['Date']= pd.to_datetime(entertainment.Date)
investment['Date']=pd.to_datetime(investment.Date)
shopping['Date']=pd.to_datetime(shopping.Date)
grocery['Date']=pd.to_datetime(grocery.Date)
transport['Date']=pd.to_datetime(transport.Date)
other['Date']=pd.to_datetime(other.Date)
home['Date']=pd.to_datetime(home.Date)


entertainment.head()
investment.head()
shopping.head()
grocery.head()
transport.head()
other.head()
#lets do some statistics
entertainment.shape

entertainment.describe()
entertainment[entertainment['amount']==entertainment['amount'].max()]

entertainment.dtypes

entertainment.isnull().any()
.
investment.isnull().any()
shopping.isnull()
#similary we can check for the other category
grocery.isnull().any()
transport.isnull().any()
other.isnull().any()
home.isnull().any()
#DATA ANALYSIS
plt.figure(figsize=(24,5))
plt.bar(entertainment['organisation'],entertainment['amount'])
#lets make it more readble and analyzable
entertainment.plot(x='organisation',y='amount',kind='barh',title='entertainment expenditure')

labels=[]
for i in entertainment['organisation']:
    labels.append(i)  
amount=[]
for i in entertainment['amount']:
    amount.append(i)
plt.pie(amount, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

plt.axis('equal')  
plt.tight_layout()
entertainment.plot(x='organisation',y='amount',kind='barh',title='entertainment expenditure')
plt.show()

fig = plt.figure()
fig.patch.set_facecolor('#B7AC9C')
# Change color of text and make a circle
plt.rcParams['text.color'] = 'black'
my_circle=plt.Circle( (0,0), 0.6,color='#B7AC9C' )
# Pieplot + circle on it
plt.pie(amount, labels=labels)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

plt.subplot(221)
plt.pie(grocery['amount'], labels=grocery['organisation'], 
autopct='%1.1f%%', shadow=True, startangle=140,)
plt.title("grocery expenditure")
plt.subplot(222)
plt.pie(shopping['amount'], labels=shopping['organisation'], 
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("shopping expenditure")
plt.subplot(223)
plt.pie(home['amount'], labels=home['organisation'], 
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("home expenditure")
plt.subplot(224)
plt.pie(transport['amount'], labels=transport['organisation'], 
autopct='%1.1f%%', shadow=True, startangle=0)
plt.title("transport expenditure")
#plt.legend(labels)
plt.show()
plt.pie(other['amount'], labels=other['organisation'], autopct='%1.1f%%', shadow=True, startangle=0)
plt.title("other expenditure")
plt.show()
plt.pie(investment['amount'], labels=investment['organisation'], autopct='%1.1f%%', shadow=True, startangle=0)
plt.title("investment expenditure")
plt.show()
#lets merge all the expenditure and save it to other csv
category=['entertainment','investment','shopping','grocery','transport','home','others']
#lets sum all the expenditure category wise
total_entertainment=entertainment['amount'].sum()
total_investment=investment['amount'].sum()
total_shopping=shopping['amount'].sum()
total_grocery=grocery['amount'].sum()
total_transport=transport['amount'].sum()
total_home=home['amount'].sum()
total_others=other['amount'].sum()
amount=[total_entertainment,total_investment,total_shopping,total_grocery,total_transport,total_home,total_others]
data={'category':category,'total':amount}
#here we created a new table which shows the total expenditure of all the category
df = pd.DataFrame(data) 
df.head(10)

plt.pie(df['total'], labels=df['category'], autopct='%1.1f%%', shadow=True, startangle=140)
plt.title(" expenditure")

df.plot(x='category',y='total',kind='barh',title='entertainment expenditure')
plt.show()
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
wedges, texts = ax.pie(df['total'], wedgeprops=dict(width=0.5), startangle=-40)
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(category[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)
plt.show()


ordered_df = df.sort_values(by='total')
data_sort=ordered_df

data_sort.plot(x='category',y='total',kind='barh',title='entertainment expenditure')
#this used to calculate the percentage  of every category
percent=[]
for i in df['total']:
    per=int((i/df['total'].sum())*100)
    percent.append(per)

percent.sort()

for i in range(len(df)):
    print("{}%  of your expenditure in {} category".format(percent[i],data_sort['category'].iloc[i]))
