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
