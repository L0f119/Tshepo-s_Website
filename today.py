from matplotlib import pyplot as plt 
import pandas as pd 
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
'''
file = pd.read_csv('Modified.csv')
df = file[:5]


app = dash.Dash()
app.layout = html.Div([
    html.H1("Hello Tshepo"),
    html.Div("Dash - A data product development from plotly"),
    dcc.Graph(
        id ='samplechart',
        figure = {
            'data' : [
                {'x': df["Speed"], 
                 'y':df["Defense"],
                 'type':'bar', 'name': 'Second Chart'},
                {'x': df["Sp. Atk"], 
                 'y':df['HP'],
                 'type':'bar', 'name': 'Third Chart'}
            ],
            'layout':{ 
                'title': 'Simple Bar Chart'
            
            
        }
        }
    )
]
    
)
if __name__ == '__main__':
    app.run_server(debug=True)
'''
app = dash.Dash()
app.layout = html.Div([
    html.H1(children = 'Hello prick',style={'textAlign':'center'}),
    html.Div('Dask till dawn'),
    dcc.Graph(
    id = 'Simple Tshepo Graph',
    figure = {'data':[
        {'x':[2,4,6], 'y':[10,12,16], 'type':'bar','name':"tshepo"},
        {'x':[3,6,9], 'y':[12,15,18], 'type':'bar'}
    ],'layout':{
        'title': 'Me',
        'plot_bgcolor':'black'
    }
             }
    )
])
if __name__ == '__main__':
    app.run_server(debug=True)
    
app2 = dash.Dash()
xs = np.random.randint(1,20,5)
ys = np.random.randint(1,20,5)

app2.layout = html.Div([
    dcc.Input(placeholder="Enter name",type='text',value=''),
    dcc.Graph(
    id ='random',
    figure={'data':[go.Scatter(x=xs,y=ys,mode='markers')],
           'layout':go.Layout(title="my scatter",xaxis={'title':'x - graph'},yaxis={'title':'y - graph'},hovermode='closest')}) 
              
])
if __name__ == '__main__':
    app2.run_server(debug=True)
    
    
app3 = dash.Dash()
app3.layout= html.Div([html.Label("Choose City bro"),
                       dcc.Dropdown(id='My dropdown',options = [{'label':'Johanesburg','value':'jhb'},
                                                                {'label':'Cape Town','value':'cpt'},
                                                                {'label':'Durban','value':'dbn'}],placeholder ='Pick'),
                       dcc.Slider(
                                  min = 1,
                                  max = 10,
                                  value = 5,
                                  marks = {i: i for i in range(10)}
              )
                      
                      ])
if __name__ == '__main__':
    app3.run_server(debug=True)
    
    
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'ID':[1,2,3,4,5],
    'Name': ['Region A','Region B','Region C','Region D','Region E'],
    'January_Rainfall': [150,220,180,np.nan,205],
    'February_Rainfall': [180,195,190,200,np.nan],
    'March_Rainfall': [210,205,220,230,190]
    
})

#df['January_Rainfall'] = df['January_Rainfall'].fillna(df['January_Rainfall'].sum())
#df['February_Rainfall'] = df['February_Rainfall'].fillna(df['February_Rainfall'].sum())
new = df[['January_Rainfall','February_Rainfall']].fillna(df[['January_Rainfall','February_Rainfall']].sum())
print(new)

q1 = new['January_Rainfall'].quantile(0.25)
q3 = new['January_Rainfall'].quantile(0.75)
qtr = q3 - q1
lower_l = q1 - 1.5*qtr
print(lower_l)
upper_l = q3 + 1.5*qtr
print(upper_l)
print(new.loc[new['January_Rainfall'] < lower_l])
print(new.loc[new['January_Rainfall'] > upper_l])

'''
q1 = read['Revenue'].quantile(0.25)
q3= read['Revenue'].quantile(0.75)
print(q1,q3)


iqr = q3 - q1
print(iqr)

lower_l = q1- 1.5 * iqr
upper_l = q3 + 1.5 * iqr
print(lower_l,upper_l)
print(read.loc[read['Revenue'] > lower_l])
'''
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# a. Create and populate the temperature_data array
temperature_data = np.array([
    [5.2, 6.3, 9.0, 12.5, 16.4, 20.1, 22.5, 22.0, 18.3, 13.5, 8.4, 5.7],  # Year 1
    [4.8, 6.0, 9.3, 13.0, 17.1, 21.0, 23.2, 22.6, 18.0, 12.8, 7.9, 4.9],  # Year 2
    [5.5, 6.8, 10.2, 13.4, 17.0, 20.8, 22.7, 21.9, 17.5, 13.1, 8.1, 5.3],  # Year 3
    [6.0, 7.2, 10.0, 14.0, 18.0, 22.0, 23.5, 23.0, 19.0, 14.0, 9.0, 6.0],  # Year 4
    [5.1, 6.4, 9.5, 12.8, 16.8, 21.2, 22.9, 22.4, 18.8, 13.7, 8.5, 5.8]   # Year 5
])

# Display the array
print("Temperature Data (in °C):")
print(temperature_data)

# b. Create a box plot using Plotly
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# `years = [f'Year {i+1}' for i in range(5)]` is a list comprehension in Python that creates a list of
# strings representing the years.
years = [f'Year {i+1}' for i in range(5)]

# Creating a DataFrame with the temperature data
# `df = pd.DataFrame(temperature_data, columns=months, index=years)` is creating a pandas DataFrame
# from the `temperature_data` NumPy array.
df = pd.DataFrame(temperature_data, columns=months, index=years)

# Melting the DataFrame for plotting
# `df.melt(var_name='Month', value_name='Temperature (°C)')` is a method call in pandas that reshapes
# the DataFrame `df` from a wide format to a long format.
df_melted = df.melt(var_name='Month', value_name='Temperature (°C)')

# Creating the box plot
# This line of code `fig_box = px.box(df_melted, x='Month', y='Temperature (°C)', title='Distribution
# of Temperatures Across Months for the Past 5 Years')` is creating a box plot using Plotly Express.
fig_box = px.box(df_melted, x='Month', y='Temperature (°C)', title='Distribution of Temperatures Across Months for the Past 5 Years')

# Display the box plot
fig_box.show()

# c. Create a heatmap using Plotly
# Creating the heatmap
# This line of code is creating a heatmap using Plotly. Here's a breakdown of the parameters:
fig_heatmap = go.Figure(data=go.Heatmap(
    z=temperature_data,
    x=months,
    y=years,
    colorscale='Viridis'
))

# Updating the layout of the heatmap
# The code `fig_heatmap.update_layout()` is used to update the layout properties of the heatmap figure
# before displaying it. Here's what each parameter is doing:
fig_heatmap.update_layout(
    title='Average Monthly Temperatures Over the Past 5 Years',
    xaxis_title='Month',
    yaxis_title='Year'
)

# Display the heatmap
fig_heatmap.show()


import pandas as pd

# The `dataSet` dictionary is defining a dataset containing information about students. Each key in
# the dictionary represents a different attribute of the students, such as 'ID', 'Name', 'Age',
# 'Math_Score', 'Science_Score', 'English_Score', and 'History_Score'. The corresponding values for
# each key are lists that store the specific data for each student.
dataSet = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [20, 21, 19, 22, 20],
    'Math_Score': [85, 95, 70, 98, 88],
    'Science_Score': [92, 88, 94, 96, 91],
    'English_Score': [78, 85, 0, 88, 90],
    'History_Score': [80, 90, 75, 92, 85]
}


# Loading the data into a DataFrame
df = pd.DataFrame(dataSet)

# Displaying the first five rows
print("Initial DataFrame:")
print(df.head())


# Checking for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Replacing 0s with the average score for each test
# The line `df.replace(0, df[['Math_Score', 'Science_Score', 'English_Score',
# 'History_Score']].mean(), inplace=True)` is replacing all occurrences of the value 0 in the
# DataFrame `df` with the mean score for each respective subject (Math, Science, English, History).
df.replace(0, df[['Math_Score', 'Science_Score', 'English_Score', 'History_Score']].mean(), inplace=True)

# Display the DataFrame after replacement
print("\nDataFrame after handling missing values:")
print(df)

# Identifying outliers using IQR
Q1 = df['Math_Score'].quantile(0.25)
Q3 = df['Math_Score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Handling outliers by capping them to the lower and upper bounds
# The line `df['Math_Score'] = df['Math_Score'].apply(lambda x: lower_bound if x < lower_bound else
# upper_bound if x > upper_bound else x)` is handling outliers in the 'Math_Score' column of the
# DataFrame `df`.
df['Math_Score'] = df['Math_Score'].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)

# Display the DataFrame after handling outliers
print("\nDataFrame after handling outliers:")
print(df)

# Filtering students who scored above 90 in both Mathematics and Science
# `filtered_df = df[(df['Math_Score'] > 90) & (df['Science_Score'] > 90)]` is filtering the DataFrame
# `df` to select rows where the 'Math_Score' is greater than 90 and the 'Science_Score' is also
# greater than 90. This operation creates a new DataFrame `filtered_df` containing only the rows where
# both conditions are met, indicating students who scored above 90 in both Mathematics and Science.
filtered_df = df[(df['Math_Score'] > 90) & (df['Science_Score'] > 90)]

# Adding a Total Score column
filtered_df['Total_Score'] = filtered_df['Math_Score'] + filtered_df['Science_Score']

# Sorting by Total Score in descending order
# `sorted_df = filtered_df.sort_values(by='Total_Score', ascending=False)` is sorting the DataFrame
# `filtered_df` based on the values in the 'Total_Score' column in descending order. This means that
# the rows in `filtered_df` will be rearranged such that the highest 'Total_Score' values will appear
# at the top of the DataFrame, and the lower values will appear towards the bottom. The
# `ascending=False` parameter specifies that the sorting should be done in descending order.
sorted_df = filtered_df.sort_values(by='Total_Score', ascending=False)

# Displaying the first five rows
print("\nFiltered and sorted DataFrame:")
print(sorted_df.head())

# Grouping by 'Age' and calculating the median score for each subject
# `median_scores = df.groupby('Age')[['Math_Score', 'Science_Score', 'English_Score',
# 'History_Score']].median()` is grouping the DataFrame `df` by the 'Age' column and then calculating
# the median score for each subject ('Math_Score', 'Science_Score', 'English_Score', 'History_Score')
# within each age group.
median_scores = df.groupby('Age')[['Math_Score', 'Science_Score', 'English_Score', 'History_Score']].median()

# Displaying the median scores
print("\nMedian scores by age:")
print(median_scores)


import numpy as np 
'''
a = np.array([1,2,3])
print(a)
b = np.array([[1,2,3,4,5],[2,4,6,8,10]])
print(b)
print(b.ndim)
print(b.shape)
print(a.dtype)
print(a.itemsize)
print(a.size)
'''
x = np.array([[2,4,6,8,10,12],[3,6,9,12,15,18]])
#print(x[0,2])
#print(x[0,:])
#print(x[:,2])
#print(x[1,0:6:2])
#x[:,3] = [1,2]
'''
y = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(y[1][0][0])
zeros = np.zeros((3,3))
print(zeros)

ones = np.ones((4,2,2),dtype='int32')
print(ones)

ful = np.full((2,3), 7, dtype='float')
print(ful)
ful2 = np.full_like(x, 7, dtype='float')
print(ful2)
'''
nums = np.random.randint(7, size=(1,10))
#print(nums)

math511 = np.identity(5)
#print(math511)

arr = np.array([[1,2,3]])
'''
r1 = np.repeat(arr,3,axis=0)
print(r1)

my_matrix = np.ones((5,5))
my_matrix[1:4,1:4] = 0
my_matrix[2,2] = 9
print(my_matrix)
'''

before = np.array([[1,2,3,4],
                   [5,6,7,8]])
print(before)
print('After: ')
after = before.reshape((4,2))
print(after)
'''
v1 = np.array([2,4,6,8])
v2 = np.array([1,3,5,7])
print(np.vstack([v1,v2,v1,v2]))
print(np.hstack([v2,v1]))
'''
print(np.transpose(before))
