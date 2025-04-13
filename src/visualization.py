import plotly.express as px

def plot_transaction_distribution(df, col='Transaction_Amount', nbins=20):
    """
    Plot distribution histogram of transaction amounts
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
    col : str
        Column name to plot
    nbins : int
        Number of bins for histogram
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Histogram figure
    """
    fig = px.histogram(df, x=col, nbins=nbins,
                      title=f'Distribution of {col}')
    return fig

def plot_transaction_by_account_type(df):
    """
    Box plot of transaction amount by account type
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Box plot figure
    """
    fig = px.box(df, x='Account_Type', y='Transaction_Amount',
                title='Transaction Amount by Account Type')
    return fig

def plot_amount_by_age(df):
    """
    Scatter plot of average transaction amount vs age
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter plot figure
    """
    fig = px.scatter(df, x='Age', y='Average_Transaction_Amount',
                    color='Account_Type',
                    title='Average Transaction Amount vs. Age',
                    trendline='ols')
    return fig

def plot_transactions_by_day(df):
    """
    Bar chart of transactions by day of week
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    fig = px.bar(df, x='Day_of_Week',
                title='Count of Transactions by Day of the Week')
    return fig

def plot_correlation_heatmap(df):
    """
    Correlation heatmap of numerical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Heatmap figure
    """
    correlation_matrix = df.corr()
    fig = px.imshow(correlation_matrix, title='Correlation Heatmap')
    return fig

def plot_anomalies(df):
    """
    Scatter plot highlighting anomalies
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data with Is_Anomaly column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter plot with anomalies highlighted
    """
    fig = px.scatter(df, x='Transaction_Amount', y='Average_Transaction_Amount',
                    color='Is_Anomaly', title='Anomalies in Transaction Amount')
    fig.update_traces(marker=dict(size=12), 
                    selector=dict(mode='markers', marker_size=1))
    return fig