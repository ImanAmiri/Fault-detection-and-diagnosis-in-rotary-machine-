import matplotlib.pyplot as plt
import pickle 
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_squared_error as MSE

import numpy as np
from scipy.stats import multivariate_normal
# from imblearn.over_sampling import SMOTE, ADASYN
import itertools

from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping


################################
from NN_models import WV_AE3
################################


class Online_Clustering:

    NAME = "Online Hierarchical Clustering"

    def __init__(
        self,
        num_harmonics_optimizer=5,
        ### Clustering Hyperparameters
        min_distance=None,
        min_samples_cluster=5,
        ### Threshold Hyperparameters
        num_samples_fitting=15,
        size_buffer_past_analysis=100,
        epochs=100,
        batch_size=32,
        threshold_prob = 1,  # [%] Percentage
        sigma_threshold=5,
        percentile_normal=99,
        compression_ratio=4,
        use_ae_embedding=True,
        train_model=True,
        adaptive_training=True,
        path_load_model="Models_test1\model_ae_custom_epoch_99.h5"
    ):
        ### Threhold Hyperparameters
        self.threshold_prob = threshold_prob/100
        self.sigma_threshold = sigma_threshold
        self.num_samples_fitting = num_samples_fitting
        self.size_buffer_past_analysis = size_buffer_past_analysis
        self.percentile_normal = percentile_normal
        ### General Clustering Hyperparameters
        self.subcluster_similarity_threshold = 0.01

        ### Autoencoder Hyperparameters
        self.use_ae_embedding = use_ae_embedding
        self.compression_ratio = compression_ratio
        self.flag_train_model = train_model
        self.flag_adaptive_training = adaptive_training
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_harmonics_optimizer = num_harmonics_optimizer
        self.path_load_model = path_load_model

        ### Other variables
        self.counter_sample = 0
        self.all_data_train = list()
        self.anomaly_score = list()
        self.mean_clusters = list()
        self.num_samples_each_cluster = list()
        self.outliers = list()
        self.cluster_id_counter = 0
        self.subcluster_id_counter = 0
        self.flag_operation_phase = False
        
        self.buffer_past_samples = list()
        self.clusters_distributions = list()

        self.min_distance = min_distance
        self.min_samples_cluster = min_samples_cluster 

    def run(self, sample):

        ### For the first n samples, add them to the training_set
        if self.counter_sample < self.num_samples_fitting:
            self.all_data_train.append(sample)
            cluster_id = 0
        ### When the list is full, fit the clustering model and get some important information
        elif self.counter_sample == self.num_samples_fitting:
            if self.use_ae_embedding:
                self.fit_ae_model()
                self.features_normal = self.encoder_model.predict(self.all_data_train)
            else:
                self.features_normal = np.array(self.all_data_train)
                del(self.all_data_train)
            self.fit_normal_data()
            self.flag_operation_phase = True

        if self.flag_operation_phase:
            cluster_id = self.operation_phase(sample)  

        self.counter_sample += 1
        return cluster_id

    def fit_ae_model(self):

        self.all_data_train = np.array(self.all_data_train)

        ### Normalize data        
        self.num_channels = self.all_data_train.shape[2]
        self.min_values = [None]*self.num_channels
        self.max_values = [None]*self.num_channels
        for channel_index in range(self.num_channels):
            self.min_values[channel_index] = np.min(self.all_data_train[:,:,channel_index])
            self.max_values[channel_index] = np.max(self.all_data_train[:,:,channel_index])
            self.all_data_train[:,:,channel_index] = self.normalize(
                self.all_data_train[:,:,channel_index],
                self.min_values[channel_index],
                self.max_values[channel_index],
            )
    
        if self.flag_train_model:
            split_ratio = 0.8
            num_sampes_training = int(len(self.all_data_train)*split_ratio)
            training_data = self.all_data_train[:num_sampes_training]
            validation_data = self.all_data_train[num_sampes_training:]

            self.fit_model(training_data,validation_data)
        else:
            self.load_models()            
        # self.plot_reconstruction_results(self.all_data_train[:1])

    def fit_normal_data(self):
                
        self.no_dimensions = self.features_normal.shape[1]

        no_normal_samples = self.features_normal.shape[0]

        mean_normal = np.zeros(self.no_dimensions)
        
        self.min_normal_data = np.min(self.features_normal,axis=0).copy()
        self.max_normal_data = np.max(self.features_normal,axis=0).copy()

        # """
        ### Normalize each feature dimension individually (if not using AE nn)
        for dimension in range(self.no_dimensions):
            if not self.use_ae_embedding:
                # self.features_normal[:, dimension] = self.normalize(
                #     self.features_normal[:, dimension],
                #     self.min_normal_data[dimension],
                #     self.max_normal_data[dimension],
                # )
                pass
            mean_normal[dimension] = np.mean(
                self.features_normal[:, dimension]
            )
        # """
        
        ### Compute covariance matrix of the normal distribution
        cov_matrix_normal = np.cov(self.features_normal.T)

        ### Create the normal distribution object
        self.normal_distribution = Mult_Gauss_Distributions(
            mean=mean_normal,
            cov_matrix=cov_matrix_normal,
            no_points=no_normal_samples,
        )

        ### Append all the normal samples to the circular buffer
        # Limit the buffer to the predefined buffer size
        self.buffer_past_samples = [element for element in self.features_normal[-self.size_buffer_past_analysis:]]
        self.buffer_clusters_labels = [0]*no_normal_samples
        self.buffer_subclusters_id = [0]*no_normal_samples
        self.buffer_clusters_labels = self.buffer_clusters_labels[-self.size_buffer_past_analysis:]
        self.buffer_subclusters_id = self.buffer_subclusters_id[-self.size_buffer_past_analysis:]

        if self.min_distance is None:
            self.min_distance = self.compute_min_distance()

        self.all_subclusters_id = [0]*no_normal_samples

    def operation_phase(self, sample):

        ### Preprocess input features
        if self.use_ae_embedding:
            sample = self.normalize_sample(sample).reshape(1,-1,self.num_channels)
            input_features = self.encoder_model.predict(sample,verbose=0)[0,:,0]
        else:
            # input_features = self.normalize_sample(sample)
            input_features = sample

        if self.check_normality(input_features):
            cluster_id = 0  # First Cluster (Normal data) 
            subcluster_id = 0
            # Update normal multivariate gaussian distribution
            # self.update_normal_distribution(input_features)       
        else:
            cluster_id, subcluster_id = self.scan_neighbourhood(input_features)
            # probs_distributions = self.compute_prob_each_distribution(input_features)
            # cluster_id = self.cluster_selection_decision(input_features,probs_distributions)
            
        self.update_circular_buffers(input_features, cluster_id, subcluster_id)         

        return cluster_id

    def standardize_sample(self, sample):
        ### Stardardize sample according to train set mean and std
        sample = np.array(
            [
                self.standardize(
                    sample[dimension_index],
                    mean=self.mean_normal[dimension_index],
                    std=self.std_normal[dimension_index],
                )
                for dimension_index in range(self.no_dimensions)
            ]
        )
        return sample

    def normalize_sample(self, sample):
        ### Normalize sample according to normal data
        sample = np.array(
            [
                self.normalize(
                    sample[dimension],
                    self.min_normal_data[dimension],
                    self.max_normal_data[dimension],
                )
                for dimension in range(self.no_dimensions)
            ]
        )
        return sample

    def check_normality(self, sample):
        return multivariate_normal.pdf(
            sample,
            mean=self.normal_distribution.mean,
            cov=self.normal_distribution.cov_matrix,
            ) > self.threshold_prob 

    def update_normal_distribution_old(self,sample):
        for dimension in range(self.no_dimensions): 
            self.normal_distribution.mean[dimension] = (
                sample[dimension] + self.normal_distribution.mean[dimension] * self.normal_distribution.no_points
                )/(self.normal_distribution.no_points+1)
            ### TODO: Implement covariance matrix's real-time update
        self.normal_distribution.no_points += 1
    
    def update_cluster_distribution_old(self,sample,cluster_id):
        for dimension in range(self.no_dimensions): 
            self.clusters_distributions[cluster_id].mean[dimension] = (
                sample[dimension] + self.clusters_distributions[cluster_id].mean[dimension] * self.clusters_distributions[cluster_id].no_points
                )/(self.clusters_distributions[cluster_id].no_points+1)
            ### TODO: Implement covariance matrix real-time update
        self.clusters_distributions[cluster_id].no_points += 1

    def update_normal_distribution(self,sample): 
        self.normal_distribution.mean = self.normal_distribution.mean + (1/(self.normal_distribution.no_points+1))*(sample-self.normal_distribution.mean)
        # self.normal_distribution.cov_matrix = self.normal_distribution.cov_matrix \
        self.normal_distribution.cov_matrix = (self.normal_distribution.cov_matrix * (self.normal_distribution.no_points/(self.normal_distribution.no_points+1))) \
            + (self.normal_distribution.no_points/(self.normal_distribution.no_points+1)**2) * np.dot(
            (sample-self.normal_distribution.mean).reshape(-1,1),(sample-self.normal_distribution.mean).reshape(1,-1)
            )
        self.normal_distribution.no_points += 1

    def update_cluster_distribution(self,sample,cluster_id):
        self.clusters_distributions[cluster_id].mean = self.clusters_distributions[cluster_id].mean \
            + (1/(self.clusters_distributions[cluster_id].no_points+1))*(sample-self.clusters_distributions[cluster_id].mean)
        # self.clusters_distributions[cluster_id].cov_matrix = self.clusters_distributions[cluster_id].cov_matrix \
        self.clusters_distributions[cluster_id].cov_matrix = (self.clusters_distributions[cluster_id].cov_matrix * (self.clusters_distributions[cluster_id].no_points/(self.clusters_distributions[cluster_id].no_points+1))) \
            + (self.clusters_distributions[cluster_id].no_points/(self.clusters_distributions[cluster_id].no_points+1)**2) * np.dot(
            (sample-self.clusters_distributions[cluster_id].mean).reshape(-1,1),(sample-self.clusters_distributions[cluster_id].mean).reshape(1,-1)
            )
        self.clusters_distributions[cluster_id].no_points += 1

    def scan_neighbourhood(self,sample):

        distances = self.compute_distances_sample_all_points(sample)

        # If this point is not close enough to any other point        
        if min(distances) > self.min_distance:
            # Check if it fits in any previously saved distributions
            probs = self.compute_prob_each_distribution(sample)
            if max(probs) > self.threshold_prob:
                cluster_id = np.argmax(probs)
                self.update_cluster_distribution(sample,cluster_id)
                subcluster_id = 0
            else:
                cluster_id = -1
                self.subcluster_id_counter += 1
                subcluster_id = self.subcluster_id_counter                
                # Limit the subcluster_id to the size of the circular buffer
                # because we are only anylizing the last n points (including outliers)
                if self.subcluster_id_counter > self.size_buffer_past_analysis:
                    self.subcluster_id_counter = 1
        # Otherwise, assign the sample to the closest island (cluster)
        # and update the cluster's distribution parameters
        else:
            # Get information from the closest cluster/point
            cluster_id = self.buffer_clusters_labels[np.argmin(distances)]
            # if cluster_id > 0: cluster_id -= 1 
            # subcluster_id = 0

            # If the closest point is an outlier, gather these points and 
            # check if there are enough outliers in the subcluster and if they can form an island  
            if cluster_id == -1:
                subcluster_id = self.buffer_subclusters_id[np.argmin(distances)]
                no_pts_subcluster = self.buffer_subclusters_id.count(subcluster_id)+1  # +1 because I'm adding the current sample
                if no_pts_subcluster >= self.min_samples_cluster:
                    self.cluster_id_counter += 1
                    self.gather_outliers_in_new_cluster(sample,subcluster_id,no_pts_subcluster)
                    cluster_id = self.cluster_id_counter
                    subcluster_id = 0
            # If the closest point belongs to a cluster, add this point to the cluster
            # and update the cluster's distribution parameters
            elif cluster_id == 0:   # Normal cluster
                self.update_normal_distribution(sample)   
                subcluster_id = 0             
            elif cluster_id > 0:    # Other clusters
                self.update_cluster_distribution(sample,cluster_id-1)                
                subcluster_id = 0

        # # # # Add 1 because cluster #0 is the normal condition
        # # # cluster_id += 1

        return cluster_id, subcluster_id

    def update_circular_buffers(self, sample, cluster_id, subcluster_id):
        self.buffer_past_samples.append(sample)
        self.buffer_clusters_labels.append(cluster_id)
        self.buffer_subclusters_id.append(subcluster_id)
        if len(self.buffer_past_samples) > self.size_buffer_past_analysis:
            self.buffer_past_samples.pop(0)
            self.buffer_clusters_labels.pop(0)
            self.buffer_subclusters_id.pop(0)

        self.all_subclusters_id.append(subcluster_id)

    def gather_outliers_in_new_cluster(self,sample,subcluster_id,no_pts_subcluster):
        ### Create a new cluster        
        indexes = np.where(np.array(self.buffer_subclusters_id) == subcluster_id)[0]
        cluster_data = np.array(self.buffer_past_samples)[indexes]
        cluster_data = np.concatenate((
            cluster_data,sample.reshape(1,-1)
        ),axis=0)
        cluster_data = self.data_oversampling(cluster_data)
        self.clusters_distributions.append(
            Mult_Gauss_Distributions(
                mean=np.mean(cluster_data,axis=0),
                cov_matrix=np.cov(cluster_data.T),
                no_points=no_pts_subcluster
            ))
        ### Update the subcluster_id buffer from old values to zeros
        self.buffer_subclusters_id = list(map(
            lambda x: 0 if x == subcluster_id else x,self.buffer_subclusters_id
            ))
        ### Update the cluster_id buffer from -1 values to new cluster id
        self.buffer_clusters_labels = [self.cluster_id_counter if i in indexes else element for i, element in enumerate(self.buffer_clusters_labels)]

    def compute_prob_each_distribution(self, sample):
        probs = list()
        ### If there are no clusters yet, return a probability lower than the threshold
        if len(self.clusters_distributions) == 0:
            probs.append(self.threshold_prob*0.9)
        else:
            for distribution in self.clusters_distributions:
                probs.append(
                    multivariate_normal.pdf(
                        sample,
                        distribution.mean,
                        distribution.cov_matrix,
                        )
                )
        return probs
    
    def data_oversampling(self,data,reps=2):
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:,0],data[:,1],data[:,2])
        mean = np.mean(data,axis=0)
        ax.scatter(mean[0],mean[1],mean[2])
        print(mean)
        plt.show()
        print(np.cov(data.T))
        """              
        for _ in range(reps): 
            new_data = list()
            set_indexes = np.arange(data.shape[0])
            for subset in itertools.combinations(set_indexes, 2):
                new_data.append(
                    np.mean([
                    data[subset[0]],
                    data[subset[1]]
                    ],axis=0
                    ))
            new_data = np.array(new_data)
            data = np.concatenate((
                data,new_data
            ),axis=0)    

        return data

    def compute_min_distance(self):

        # Compute distance of all 5 closest neighbours
        min_distances = list()
        max_distances = list()
        no_normal_samples = self.features_normal.shape[0]
        for sample in range(no_normal_samples):
            distances = np.array(self.compute_distances_sample_all_points(self.features_normal[sample]))
            # distances = np.sort(distances)
            # min_distances.append(distances[5])
            max_distances.append(max(distances))

        # The min_distance parameter will be defined based on
        # Mean + 5 x Std
        # dist1 = np.mean(min_distances) + 5*np.std(min_distances)
        dist2 = np.mean(max_distances) + 5*np.std(max_distances)

        return dist2

    def compute_distances_with_centroids(self, sample):
        distances = list()
        ### If there are no clusters yet, the first cluster will be the anomaly sample
        if len(self.mean_clusters) == 0:

            self.mean_clusters.append(sample)
            self.num_samples_each_cluster.append(1)
        for element in self.mean_clusters:
            distances.append(
                # self.cosine_similarity(sample, self.mean_clusters[i])
                np.linalg.norm(sample - element)
            )
        return distances

    def compute_distances_sample_all_points(self, sample):
        distances = list()
        for point in self.buffer_past_samples:
            distances.append(
                np.linalg.norm(sample - point)
            )
        return distances

    def compute_distances_point_outliers(self, sample):
        distances = list()
        for outlier in self.outliers:
            distances.append(
                np.linalg.norm(sample - outlier)
            )
        return distances

    def cluster_selection_decision(self,sample,probabilities):
        ### Get distribution with highest probability 
        highest_prob = np.max(probabilities)

        ### If the sample is likely enough to be in one distribution
        # it will be included to the cluster, and the distribution's mean and covariance matrix will be updated
        if highest_prob >= self.threshold_prob:
            cluster_id = np.argmin(probabilities)
            self.update_cluster_distribution(sample,cluster_id)    
            cluster_label = cluster_id + 1  
        # elif minimun_distance < self.second_threshold:
        #     cluster_id = np.argmin(distances_centroids)
        #     self.num_samples_each_cluster[cluster_id] += 1
        #     cluster_label = cluster_id + 1 
        ### If all the probabilities are too small, a new cluster will be created
        else:
            # If it is the first outlier
            if len(self.outliers) == 0:
                self.outliers.append(sample) 
                cluster_label = -1
            else:
                distances_outliers = self.compute_distances_point_outliers(sample)
                minimun_distance = np.min(distances_outliers)
                ### Check if the outlier can gather with another outlier
                if minimun_distance < self.thr_distance_normal:  
                    closest_outlier_id = np.argmin(distances_outliers)
                    new_centroid = self.update_centroid(
                        sample,
                        self.outliers[closest_outlier_id],
                        1,
                        )              
                    self.mean_clusters.append(new_centroid)
                    self.num_samples_each_cluster.append(2)
                    # Remove outlier from the outlier list 
                    # (as it was assigned a new cluster now)
                    self.outliers.pop(closest_outlier_id)
                    cluster_label = len(self.mean_clusters)+1
                else:
                    self.outliers.append(sample)
                    cluster_label = -1
                         
        return cluster_label

    def update_centroid(self, sample, centroid, num_samples_cluster):
        new_centroid = np.zeros(self.no_dimensions)
        for dimension in range(self.no_dimensions): 
            new_centroid[dimension] = (sample[dimension] + centroid[dimension]*(num_samples_cluster))/(num_samples_cluster+1)
        return new_centroid

    def cosine_similarity(self, sample, centroid):
        return np.dot(sample, centroid) / (
            np.linalg.norm(sample) * np.linalg.norm(centroid)
        )

    def standardize(self, data, mean, std):
        return (data - mean) / std

    def normalize(self, data, min, max):
        return (data - min)/(max - min)

    def fit_model(self,training_data, validation_data):

        save_model_each_epoch = MyCallback()
        early_stopping = [
            EarlyStopping(
                monitor="mse",
                min_delta=1e-5,
                patience=20,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )
            ]

        if not self.flag_adaptive_training:
            ##############################################################################
            ### First of all, train the model with only MSE of time domain untill stopping by earlystopping       
            wv_model = self.initialize_model(training_data,loss_function="mse")
            self.ae_model = wv_model.ae_model
            self.encoder_model = wv_model.encoder_model
            self.decoder_model = wv_model.decoder_model
                           
            self.ae_model.fit(
                    x=training_data,
                    y=training_data,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(validation_data, validation_data),
                    callbacks=[save_model_each_epoch,early_stopping],
                    verbose=1,
                )
            ### Save models 
            self.ae_model.save("model_ae_MSE.h5")
            self.decoder_model.save("model_encoder_MSE.h5")
            self.decoder_model.save("model_decoder_MSE.h5")

        ##############################################################################
        ### Then, train the model with the dynamic loss function
        else:
            wv_model = self.initialize_model(training_data,loss_function="mse")
            # """
            wv_model.recompile_model()
            self.ae_model = wv_model.ae_model
            self.encoder_model = wv_model.encoder_model
            self.decoder_model = wv_model.decoder_model
            history = self.ae_model.fit(
                x=training_data,
                y=training_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(validation_data, validation_data),
                callbacks=[save_model_each_epoch, early_stopping],
                verbose=1,
            )
            self.ae_model.save("Models/model_ae_CUSTOM.h5")
            self.encoder_model.save("Models/model_encoder_CUSTOM.h5")
            self.decoder_model.save("Models/model_decoder_CUSTOM.h5")
            
            with open("Models/fit_history","w") as file:
                pickle.dump(history.history,file) 

    def initialize_model(self,training_data,loss_function="mse"):
        return WV_AE3(
                    input_dimension=training_data.shape[1],
                    batch_size=self.batch_size,
                    num_sensors=1,
                    training_data=training_data,
                    num_harmonics_optimizer=self.num_harmonics_optimizer,
                    ### Hiperparameters
                    compression_ratio=self.compression_ratio,
                    loss_function=loss_function,
                    )

    def plot_reconstruction_results(self, training_data):
        sample = training_data
        sample_rec = self.ae_model.predict(sample)

        figure = Figure()
        figure.plot(sample[0,:,0], sample_rec[0,:,0])
        figure.show("Models/Reconstruction_custom_loss")
    
    def load_models(self):
        self.ae_model = load_model(self.path_load_model,compile=False)
        self.encoder_model = self.ae_model.layers[1]
        self.decoder_model = self.ae_model.layers[2]

###########################################
# Tensorflow Callbacks
###########################################

def compare_weights(w1,w2):
    error = 0
    for i in range(len(w1)):
        error =+ MSE(w1[i].flatten(),w2[i].flatten())
    return error

class Mult_Gauss_Distributions():
    def __init__(
        self,
        mean,
        cov_matrix, 
        no_points=1,    #Initialize with one single point 
    ):
        self.mean = mean
        self.cov_matrix = cov_matrix        
        self.no_points = no_points      

class MyCallback(Callback):
    def __init__(self):
        self.epoch = 0
        self.batch = 0
        self.mse_list = list()

    def on_epoch_end(self, epoch, logs=None):
        # print(f"EPOCH = {epoch}")
        self.model.save(f"Models/model_ae_custom_epoch_{epoch}.h5")
        self.epoch += 1
        if self.epoch == 1:
            np.save("Models/MSE_first_epoch",self.mse_list)
            plt.figure()
            plt.title("MSE first epoch")
            plt.plot(self.mse_list)
            plt.savefig("Models/MSE_first_epoch")

    def on_train_batch_end(self, batch, logs=None):
        if "mse" in logs.keys():
            self.mse_list.append(logs['mse'])

### For plotting using plotly library
class Figure(object):
    def __init__(self):
        self.fig = make_subplots(
            rows=1,cols=2,
            subplot_titles=("Time Domain", "Frequency Domain")            
            )
        
    def plot(self, original_signal, reconstructed_signal, Fs=None):
        if Fs is None:
            Fs = len(original_signal)
        error_mse = MSE(original_signal,reconstructed_signal)
        self.fig.update_layout(title_text=f"Normalized MSE = {error_mse:0.5f}")        
        x = np.arange(len(original_signal))        
        spectrum_orig, x_freq, _ = plt.magnitude_spectrum(original_signal, Fs=Fs)
        spectrum_rec, _, _ = plt.magnitude_spectrum(reconstructed_signal, Fs=Fs)

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=original_signal,
                name="Original",
                line_color="blue",
            ),
            row=1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=reconstructed_signal,
                name="Reconstructed",
                line_color="orange",                
            ),
            row=1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x_freq,
                y=spectrum_orig,
                name="Original",
                line_color="blue",
            ),   
            row=1,
            col=2,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x_freq,
                y=spectrum_rec,
                name="Reconstructed",
                line_color="orange",
            ),
            row=1,
            col=2,
        )

        # Update xaxis properties
        self.fig.update_xaxes(title_text="Points", row=1, col=1)
        self.fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=2)
        # Update yaxis properties
        self.fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        self.fig.update_yaxes(title_text="Energy", row=1, col=2)

    def show(self, filename):
        self.fig.show()
        plot_offline(self.fig, filename=filename+'.html')