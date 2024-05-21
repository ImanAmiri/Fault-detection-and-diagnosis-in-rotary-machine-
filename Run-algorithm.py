import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot as plot_offline 
from plotly.subplots import make_subplots
from plotly import graph_objects as go

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.metrics import mean_squared_error as rmse
from scipy.stats import kurtosis, skew
from scipy.fft import fft as FFT
from sklearn.decomposition import PCA

from online_Clustering2 import Online_Clustering

def feature_extraction(data, num_normal_samples:int, num_stats_features=3, num_harmonics=15):

    # Remove mean from the signal
    data = data[:,:,0] - np.mean(data)

    """
    aux = data[:1]
    aux = aux[:,::2,:]
    aux = np.concatenate((
        aux,aux
    ),axis=1)
    data[0] = aux
    """

    num_samples = data.shape[0]
    n = data.shape[1]

    ### Statistic features
    if num_stats_features > 3:
        num_stats_features = 3
    stats_features = ["RMS","Skewness","Kurtosis"][:num_stats_features]    

    ### Harmonic features
    # Compute harmonic peaks of the FFT spectrum
    if num_harmonics > 0:
        fft_samples = np.array([FFT(sample) for sample in data])
        fft_samples = 2.0/n * np.abs(fft_samples[:,0:n//2]) 
        x1 = find_rotation_speed(fft_samples[:num_normal_samples,:100])           

    ### 3 features (RMS, Skewness, Kurtosis)
    new_data = np.zeros((
        num_samples,
        num_stats_features + 2*num_harmonics # 2*num_harmonics beacause x,y point of FFT
        ))

    for sample_index in range(num_samples):
        count_feature = 0
        
        if "RMS" in stats_features:
            ### RMS
            zero_vector = np.zeros((data.shape[1]))
            rms = rmse(
                data[sample_index,:], zero_vector, multioutput="raw_values", squared=False
            )
            new_data[sample_index,count_feature] = rms
            count_feature += 1
        if "Skewness" in stats_features:
            ### Skewness
            skewness = skew(data[sample_index,:])
            new_data[sample_index,count_feature] = skewness
            count_feature += 1
        if "Kurtosis" in stats_features:
            ### Kurtosis
            kurt = kurtosis(data[sample_index,:])
            new_data[sample_index,count_feature] = kurt
            count_feature += 1
            
        # Find peaks 
        if num_harmonics > 0:
            freq_bound = int(x1*0.2)
            peak_points = np.array([(
                np.argmax(fft_samples[
                    sample_index,
                    (harmonic+1)*x1-freq_bound:(harmonic+1)*x1+freq_bound
                    ])+(harmonic+1)*x1-freq_bound,
                np.max(fft_samples[
                    sample_index,
                    (harmonic+1)*x1-freq_bound:(harmonic+1)*x1+freq_bound
                    ]),
                ) for harmonic in range(num_harmonics)])
            peak_points = peak_points.reshape(-1)

            new_data[sample_index,count_feature:] = peak_points

    new_data = new_data[:,[0,2,4]]
    # new_data = new_data[:,[0,1,3]]

    return new_data

    """
    total_features = new_data.shape[1]
    num_rows = int(total_features/3)
    subplot_list = list()

    plt.figure()
    plt.plot(data.reshape(-1))
    plt.show()

    plt.figure()
    subplot_list.append(plt.subplot(num_rows,3,1))
    subplot_list[0].plot(new_data[:,0])
    subplot_list[0].set_ylabel("RMS")
    subplot_list.append(plt.subplot(num_rows,3,2,sharex=subplot_list[0]))
    subplot_list[1].plot(new_data[:,1])
    subplot_list[1].set_ylabel("Skewness")
    subplot_list.append(plt.subplot(num_rows,3,3,sharex=subplot_list[0]))
    subplot_list[2].plot(new_data[:,2])
    subplot_list[2].set_ylabel("Kurtosis")
    for subplot_index in range(3,total_features):
        subplot_list.append(plt.subplot(num_rows,3,subplot_index,sharex=subplot_list[0]))
        subplot_list[subplot_index].plot(new_data[:,subplot_index])

    plt.show()
    """
    
def find_rotation_speed(fft_signals):        
    rotation_speeds = np.argmax(fft_signals,axis=1)
    return int(np.mean(rotation_speeds))

def stardardization(data, mean, std):
    no_dimensions = data.shape[1]
    for dimension in range(no_dimensions):
        data[:,dimension] = (data[:,dimension] - mean[dimension])/std[dimension]
    return data

def plot_results2(data, answers, targets, subclusters_id, online_clust_obj, subsampling_rate=10):
    
    answers_all_points = list()
    targets_all_points = list()
    subclusters_all_points = list()
    num_samples = data.shape[0]
    sample_dimension = data.shape[1]
    for i in range(num_samples):
        answers_all_points.append(np.ones(sample_dimension)*answers[i])
        targets_all_points.append(np.ones(sample_dimension)*targets[i])
        subclusters_all_points.append(np.ones(sample_dimension)*subclusters_id[i])
    answers_all_points = np.array(answers_all_points).reshape(-1)
    targets_all_points = np.array(targets_all_points).reshape(-1)
    subclusters_all_points = np.array(subclusters_all_points).reshape(-1)

    ########################################
    ### Signal with the targets and clusters
    fig = make_subplots(
        rows=4,cols=1,
        shared_xaxes=True,
        # subplot_titles=("Signal Time", "Targets", "Cluster IDs")            
        )
    data_plot = data.reshape(-1)[::subsampling_rate]
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(data_plot)),
                y=data_plot,
                name="Signal",
                line_color="blue",
                line_width=1,
            ),
            row=1,
            col=1,
        )
    targets_plot = targets_all_points[::subsampling_rate]
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(targets_plot)),
                y=targets_plot,
                name="Targets",
                line_color="red",
            ),
            row=2,
            col=1,
        )
    answers_plot = answers_all_points[::subsampling_rate]
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(answers_plot)),
                y=answers_plot,
                name="Cluster ID's",
                line_color="green",
            ),
            row=3,
            col=1,
        )
    subcluster_plot = subclusters_all_points[::subsampling_rate]
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(subcluster_plot)),
                y=subcluster_plot,
                name="Sub_Cluster ID's",
                line_color="orange",
            ),
            row=4,
            col=1,
        )
    plot_offline(fig, filename='Signal_targets_clusters_subclustersID.html')
    fig.show()

    ### Generate log with number of occurences of each subcluster ID
    ### and another log with number of samples per cluster
    import pandas as pd
    df_subclust_id = pd.DataFrame(columns=["Subcluster_ID","Count"])
    no_subclust_ids = max(subclusters_id)
    for i in range(no_subclust_ids):
        df_subclust_id.loc[len(df_subclust_id)] = [i, subclusters_id.count(i)]

    df_clust_id = pd.DataFrame(columns=["Cluster_ID","Count"])
    no_clusters = len(online_clust_obj.clusters_distributions)
    for i in range(no_clusters):
        df_clust_id.loc[len(df_clust_id)] = [i, online_clust_obj.clusters_distributions[i].no_points]

    import os
    df_subclust_id.to_csv(os.getcwd()+"\\df_subcluster_IDs.csv",index=False)
    df_clust_id.to_csv(os.getcwd()+"\\df_no_clusters.csv",index=False)

def plot_results(data, answers, targets, features, features_norm=None):
    
    answers_all_points = list()
    targets_all_points = list()
    num_samples = data.shape[0]
    sample_dimension = data.shape[1]
    for i in range(num_samples):
        answers_all_points.append(np.ones(sample_dimension)*answers[i])
        targets_all_points.append(np.ones(sample_dimension)*targets[i])
    answers_all_points = np.array(answers_all_points).reshape(-1)
    targets_all_points = np.array(targets_all_points).reshape(-1)

    ########################################
    ### Signal with the targets and clusters
    fig = make_subplots(
        rows=3,cols=1,
        # shared_xaxes=True,
        # subplot_titles=("Signal Time", "Targets", "Cluster IDs")            
        )
    data_plot = data.reshape(-1)[::10]
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(data_plot)),
                y=data_plot,
                name="Signal",
                line_color="blue",
                line_width=1,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(targets)),
                y=targets,
                name="Targets",
                line_color="red",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
            go.Scatter(
                x=np.arange(len(answers)),
                y=answers,
                name="Cluster ID's",
                line_color="green",
            ),
            row=3,
            col=1,
        )
    plot_offline(fig, filename='Signal_targets_clusters.html')
    fig.show()

    #################################
    features_pca = features

    fig = go.Figure(data=[go.Scatter3d(
            x=features_pca[:,0], 
            y=features_pca[:,1], 
            z=features_pca[:,2], 
            mode='markers',
            marker=dict(
                size=6,
                color=answers,                
                colorscale='Viridis',   
                opacity=0.8,
                colorbar=dict(thickness=20,title='Cluster'),
                showscale=True
            ),           
    )])
    fig.update_layout(
        title='Outputs',
        scene=go.Scene(
            xaxis=go.XAxis(title='RMS'),
            yaxis=go.YAxis(title='1X'),
            zaxis=go.ZAxis(title='2X')
        )) #,showlegend=True,) #,margin=dict(l=0, r=0, b=0, t=0))
    plot_offline(fig, filename='Clusters_Outputs.html')
    fig.show()
    """
    fig = go.Figure(data=[go.Scatter3d(
            x=features_pca[:,0], 
            y=features_pca[:,1], 
            z=features_pca[:,2], 
            mode='markers',
            marker=dict(
                size=6,
                color=targets,                
                colorscale='Viridis',   
                opacity=0.8,
                colorbar=dict(thickness=20,title='Cluster'),
            )
        )])
    fig.update_layout(
        title='Targets',
        scene=go.Scene(
            xaxis=go.XAxis(title='RMS'),
            yaxis=go.YAxis(title='1X'),
            zaxis=go.ZAxis(title='2X')
        )) 
    plot_offline(fig, filename='Clusters_Targets.html')
    fig.show()
    """
    """
    fig = go.Figure(data=[go.Scatter3d(
            x=features_pca[:,0], 
            y=features_pca[:,1], 
            z=features_pca[:,2], 
            mode='markers',
            marker=dict(
                size=6,
                color=np.arange(len(targets)),                
                colorscale='Viridis',   
                opacity=0.8,
                colorbar=dict(thickness=20,title='Seconds'),
                showscale=True
            ),           
    )])
    fig.update_layout(
        title='Time sequence',
        scene=go.Scene(
            xaxis=go.XAxis(title='RMS'),
            yaxis=go.YAxis(title='1X'),
            zaxis=go.ZAxis(title='2X')
        )) #,showlegend=True,) #,margin=dict(l=0, r=0, b=0, t=0))
    plot_offline(fig, filename='Clusters_Time_sequence.html')
    fig.show()
    """

def load_data(current_test,path,number_of_channels=1):

    data = np.load(
        path + "Teste_" + str(current_test) + ".npy"
    )
    if current_test <= 5 or current_test == 10:
        targets = np.load(
                path + "Targets_teste_" + str(current_test) + ".npy" 
            )
    else:
        targets = np.ones(data.shape[0])

    if current_test == 1:
        # Fs = 4000
        Fs = 4000       
    elif current_test == 2:
        # Fs = 19638.6 
        Fs = 20000
    elif current_test == 3:
        # Fs = 16383.99
        Fs = 20000
    elif current_test == 4:
        Fs = 10000
    elif current_test == 5:
        Fs = 20480
    elif current_test == 6:
        Fs = 4096
    elif current_test == 7:
        Fs = 2048
    elif current_test == 8:
        Fs = 4096
    elif current_test == 10:
        Fs = 2048

    # Split data into one-second fragments  
    num_samples = int(data.shape[0]/Fs)
    data = data[:Fs*num_samples,:number_of_channels] 

    ### Slice into 1s fragments
    data = data.reshape((num_samples,Fs,number_of_channels))    
    # Leave only the first 128 points of each sample 
    # (to start testing clustering with median dimension data)
    
    # plt.figure(),plt.plot(data.reshape(-1))
    # plt.figure(),plt.plot(data[:,:128].reshape(-1))
    # plt.figure(),plt.plot(data[:,:128:10].reshape(-1))

    # data = data[:,:512:5]

    if current_test == 1:
        data = data[:,:2048,:]
    # data = data[:,::40,:]


    targets = targets[::Fs][:num_samples]

    # plt.figure(),plt.plot(data.reshape(-1)),plt.show()

    return data, targets

if __name__ == '__main__':

    ### Implement outlier clusters to a single label (change this label after waiting N samples)
    ### Extract features from the x,y values of the FFT peaks (DONE)

    ############
    # FIX THE LOSS FUNCTION
    ############
    
    path = "C:\\Users\\ahenri01\\OneDrive - Universidade Federal de Uberlândia\\Documents\\Venv\\Data_benchmarks - New_Targets - Online_Clustering\\"
    current_test = 10
    num_normal_samples = 100
    
    # Load data from Lab, from Petrobras or from CWRU    
    signal, targets = load_data(
        current_test=current_test,
        path=path
        )

    #################################################  
    """
    targets_copy = targets.copy()
    diffs = np.diff(targets)
    diffs = np.concatenate((np.zeros(1),diffs))  #Add first position

    ones = np.where(targets==1)
    plus_ones = np.where(diffs==1)[0]
    minus_ones = np.where(diffs==-1)[0]

    plus_ones = np.concatenate((plus_ones,np.ones(1)*len(diffs)))  #Add first position
    plus_ones = plus_ones.astype(int)

    # minus_ones = np.concatenate((np.ones(1)*0,minus_ones))  #Add first position
    minus_ones = np.concatenate((minus_ones,np.ones(1)*len(diffs)))  #Add first position
    minus_ones = minus_ones.astype(int)

    aux_counter = 0
    for i in range(len(minus_ones)):
        targets[aux_counter:minus_ones[i]] = i
        # targets[minus_ones[i]:plus_ones[i]] = i
        
        aux_counter = minus_ones[i]
        # print((i,element))
    # targets[ones] = -1
    """    
    # plt.figure()
    # plt.plot(signal.reshape(-1))
    # plt.show()
    #################################################    

    use_ae_embedding = False

    if use_ae_embedding:
        data = signal
    else:
        data = feature_extraction(
            signal,
            num_normal_samples,
            num_stats_features=1, 
            num_harmonics=2,
            )

    online_clust_obj = Online_Clustering( 
        num_samples_fitting=num_normal_samples,   
        compression_ratio=6,  ##The Autoencoder compression equals 2^compression_ratio    
        use_ae_embedding=use_ae_embedding,
        train_model=True,
        adaptive_training=True,
        epochs=100,
        num_harmonics_optimizer=5,
        size_buffer_past_analysis=200,
        # min_distance=15,
        min_samples_cluster=10,
        )
    
    """### TODO: Implement automatic min_distance value based on normal neighour distances"""
    """### TODO: Implement real-time covariance matrix update"""
    """### TODO: Get all subcluster_ids values"""
    """### TODO: Implement oversampling"""
    ### TODO: Check the outliers gathering

    ############################################################
    answers = list()
    features_norm = list()
    for sample_index in range(len(data)):
        answers.append(online_clust_obj.run(data[sample_index]))
        print(f"Sample #{online_clust_obj.counter_sample}")
    ############################################################
    
    ### Standardize data according to normal condition
    # features_norm = stardardization(data, online_clust_obj.mean_normal, online_clust_obj.std_normal)

    plot_results(signal, answers, targets, data) #, features_norm)

    subclusters_id = online_clust_obj.all_subclusters_id
    plot_results2(signal,answers,targets,subclusters_id, online_clust_obj)

    print('\n--------------- \nEnd \n---------------\n')

    test=0