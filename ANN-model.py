import numpy as np
import tensorflow as tf
from keras import backend as K
import math
from sklearn.metrics import mean_squared_error as MSE
from scipy.fft import fft as FFT


class WV_AE3:

    NAME = "Wavenet_Autoencoder_v3_0"

    def __init__(
        self,
        num_sensors: int,
        training_data,
        input_dimension=None,
        batch_size=32,
        flag_train_model=True,
        train_on_source_set=False,
        num_harmonics_optimizer=3,
        ### Hiperparameters
        compression_ratio=2,  ##The compression will be equal 2^wavenet_residual_layers
        wavenet_residual_layers=4,
        conv_filters=16,  ##Defauls:16,
        conv_kernel_size=16,  ##Defauls:2,
        activation_function="relu",
        optimizer_function="adam",
        learning_rate = 1e-1,
        loss_function="mse",
    ):
        self.input_dimension = input_dimension
        self.num_sensors = num_sensors
        self.N = training_data.shape[1]
        self.flag_train_model = flag_train_model
        self.train_on_source_set = train_on_source_set
        self.num_harmonics_optimizer = num_harmonics_optimizer

        self.batch_size = batch_size
        self.no_batches = np.ceil(training_data.shape[0] / batch_size)

        ### Hiperparameters
        self.compression_ratio = compression_ratio
        self.wavenet_residual_layers = wavenet_residual_layers
        self.conv_filters = conv_filters  ##Defauls:16,
        self.conv_kernel_size = conv_kernel_size  ##Defauls:2,
        self.conv_strides = 2  ##Fixed for the Autoencoder Architecture
        self.learning_rate = learning_rate
        if optimizer_function == "Adam":
            self.optimizer_function = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            self.optimizer_function = optimizer_function
        if activation_function == "custom":
            self.activation_function = cosines_activation_funtion
        else:
            self.activation_function = activation_function

        self.loss_function = loss_function
        if loss_function == "custom":
            self.loss_function = self.custom_loss_function

        ### =====================================
        if self.flag_train_model:
            ### Initialize Model
            self._define_model()
            if self.train_on_source_set:
                self.define_source_set(training_data)                
        else:
            ### Load Model
            self._load_model()

        self.multiplication_factors = list()
        self.weights = list()
        self.set_worst_f_values(training_data)
        self.target_obj_function = 1 * 10**-8

        #############################################
        ### Start with only X1 adjusting
        # keep_weights = [0,2]
        # self.weights = [0 if i not in keep_weights else element for i,element in enumerate(self.weights)]
        #############################################

        ### Normalize weights so that the sum equals 1
        sum_list = sum(self.weights)
        self.weights = [element / sum_list for element in self.weights]

        # self.weights = [1]*(self.num_harmonics_optimizer+2)

    def _define_model(self):

        wavenet_layers = 2 ** np.arange(self.compression_ratio)

        ### ===============================
        ### ENCODER
        ### ===============================
        encoder_input = tf.keras.layers.Input(
            shape=(self.input_dimension, self.num_sensors)
        )

        flag_fist_layer_encoder = True
        for rate in wavenet_layers:
            if flag_fist_layer_encoder:
                encoded = tf.keras.layers.Conv1D(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    padding="causal",
                    activation=self.activation_function,
                    dilation_rate=int(rate),
                )(encoder_input)
                flag_fist_layer_encoder = False
            else:
                encoded = tf.keras.layers.Conv1D(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    padding="causal",
                    activation=self.activation_function,
                    dilation_rate=int(rate),
                )(encoded)
            encoded = tf.keras.layers.Conv1D(
                filters=self.conv_filters,
                kernel_size=self.conv_kernel_size,
                strides=self.conv_strides,
                padding="same",
                activation=self.activation_function,
            )(encoded)
        encoded = tf.keras.layers.Conv1D(
            filters=self.num_sensors,
            kernel_size=1,
            name="Last_encoder",
            activation=self.activation_function,
        )(encoded)

        encoder = tf.keras.models.Model(encoder_input, encoded)

        ### ===============================
        ### DECODER
        ### ===============================
        # decoder_input = tf.keras.layers.Input(shape=(latent_dim, self.conv_filters))
        decoder_input = tf.keras.layers.Input(
            shape=(encoded.shape[1], encoded.shape[2])
        )

        flag_fist_layer_decoder = True
        for rate in wavenet_layers:
            if flag_fist_layer_decoder:
                decoded = tf.keras.layers.Conv1DTranspose(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    strides=self.conv_strides,
                    padding="same",
                    activation=self.activation_function,
                )(decoder_input)
                flag_fist_layer_decoder = False
            else:
                decoded = tf.keras.layers.Conv1DTranspose(
                    filters=self.conv_filters,
                    kernel_size=self.conv_kernel_size,
                    strides=self.conv_strides,
                    padding="same",
                    activation=self.activation_function,
                )(decoded)
            decoded = tf.keras.layers.Conv1D(
                filters=self.conv_filters,
                kernel_size=self.conv_kernel_size,
                padding="causal",
                activation=self.activation_function,
                dilation_rate=int(rate),
            )(decoded)

        output_layer = tf.keras.layers.Conv1D(
            self.num_sensors,
            1,
            name="Last_conv_layer",
            # activation=self.activation_function,
            activation="tanh",
        )(decoded)

        decoder = tf.keras.models.Model(decoder_input, output_layer)

        ### ===============================
        ### AUTOENCODER
        ### ===============================
        ae_input = tf.keras.layers.Input(shape=(self.input_dimension, self.num_sensors))

        encoded = encoder(ae_input)
        decoded = decoder(encoded)

        ae_model = tf.keras.models.Model(ae_input, decoded)

        ae_model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer_function,
            metrics="mse",
        )
        # ae_model.compile(
        #     loss="mse", optimizer=self.optimizer_function
        # )

        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nEncoder\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # encoder.summary()
        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nDecoder\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # decoder.summary()
        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nAutoencoder\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        ae_model.summary()

        self.ae_model = ae_model
        self.encoder_model = encoder
        self.decoder_model = decoder

    def define_source_set(self,training_data,no_harmonics=15,bootstrap_ratio=100):
        """
        Generates syntetic data based on the frequency spectrum of the original data
        """

        fft_training_data = FFT(training_data.reshape(-1))
        fft_training_data = 2.0 / self.N * np.abs(fft_training_data[0 : self.N // 2])

        import matplotlib.pyplot as plt
        spectrum_orig, x_freq, _ = plt.magnitude_spectrum(training_data.reshape(-1),Fs=training_data.shape[1])

        for sample in training_data:
            pass

            
            self.find_harmonics




    def _load_model(self):
        self.encoder_model = tf.keras.models.load_model("models\\model_encoder.h5")
        self.decoder_model = tf.keras.models.load_model("models\\model_decoder.h5")
        self.ae_model = tf.keras.models.load_model("models\\model_ae.h5")

        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nEncoder\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # self.encoder_model.summary()
        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nDecoder\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # self.decoder_model.summary()
        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nAutoencoder\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        self.ae_model.summary()

    def set_worst_f_values(self, training_data):

        training_data = training_data[:, :, 0]
        training_data -= np.mean(training_data)

        ### Loss 1 (Time domain)
        multiplication_factor = self.compute_multiplication_factor(training_data)
        self.multiplication_factors.append(multiplication_factor)

        self.worst_f_value_time = self.compute_mean_error(
            training_data * multiplication_factor
        )
        self.weights.append(1)

        ### Loss 2 and so on (Frequency domain fragments)
        if self.num_harmonics_optimizer > 0:
            self.worst_f_values_freq = np.zeros(self.num_harmonics_optimizer + 1)

            ### Compute FFT of each sample
            fft_samples = FFT(training_data)
            fft_samples = 2.0 / self.N * np.abs(fft_samples[:, 0 : self.N // 2])

            self.rotation_speed = self.find_rotation_speed(fft_samples)
            nodes = [0, 0.5]
            self.freq_bounds = [
                nodes[0] * self.rotation_speed,
                nodes[1] * self.rotation_speed,
            ]
            self.freq_bounds = [
                int(i) for i in self.freq_bounds
            ]  # Convert values to only integers

            freq_fragment = fft_samples[:, self.freq_bounds[0] : self.freq_bounds[1]]
            multiplication_factor = self.compute_multiplication_factor(freq_fragment)
            self.multiplication_factors.append(multiplication_factor)
            freq_fragment *= multiplication_factor

            self.worst_f_values_freq[0] = self.compute_mean_error(freq_fragment)
            self.weights.append(1)
            ### Remove the first component (0.5X) from the loss function
            # self.weights[-1] = 0

            for i in range(1, self.num_harmonics_optimizer + 1):
                nodes.append(i + 0.5)
                self.freq_bounds.append(int(nodes[-1] * self.rotation_speed))

                freq_fragment = fft_samples[
                    :, self.freq_bounds[i] : self.freq_bounds[i + 1]
                ]
                multiplication_factor = self.compute_multiplication_factor(
                    freq_fragment
                )
                self.multiplication_factors.append(multiplication_factor)
                freq_fragment *= multiplication_factor

                self.worst_f_values_freq[i] = self.compute_mean_error(freq_fragment)
                self.weights.append(1)
        test = 0

    def normalize(self, data, min, max):
        return (data - min) / (max - min)

    def compute_mean_error(self, data):

        zeros_vector = np.zeros(data.shape)
        mse_errors = MSE(
            data.T,
            zeros_vector.T,
            multioutput="raw_values",
        )
        return np.mean(mse_errors)

    def compute_multiplication_factor(self, data):

        return 1 / np.mean(np.max(data, axis=1))

    def compute_multiplication_factor_tf(self, data):

        return 1 / tf.math.reduce_mean(tf.math.reduce_max(data, axis=1))

    def find_rotation_speed(self, fft_samples):        
        # rotation_speeds = np.argmax(fft_samples, axis=1)
        rotation_speeds = np.argmax(fft_samples[:,:100], axis=1)
        return int(np.mean(rotation_speeds))

    def recompile_model(self):
        self.ae_model.compile(
            loss=self.custom_loss_function,
            optimizer=self.optimizer_function,
            metrics="mse",
            run_eagerly=True,
        )

    def find_peak_x(self, data):
        
        return tf.cast(
            (tf.math.reduce_mean(tf.math.argmax(data,axis=1))),
            tf.float32)

    def find_peak_y(self, data):

        return tf.cast(
            (tf.math.reduce_mean(tf.math.reduce_max(data,axis=1))),
            tf.float32)

    def dynamically_adjust_weights(self,fft_true,fft_pred):
        
        weights = list()

        ### Get FFT fragments according to each harmonic        
        fragment_true = fft_true[:, self.freq_bounds[0] : self.freq_bounds[1], :]
        fragment_pred = fft_pred[:, self.freq_bounds[0] : self.freq_bounds[1], :]

        peak_true = (self.find_peak_x(fragment_true), self.find_peak_y(fragment_true))
        peak_pred = (self.find_peak_x(fragment_pred), self.find_peak_y(fragment_pred))

        # Formula considering x and y of peak point 
        # error_peaks = ((peak_true[0]-peak_pred[0])/peak_true[0]) + ((peak_true[1]-peak_pred[1])/peak_true[1])
        error_peaks = ((peak_true[0]-peak_pred[0])**2 + (peak_true[1]-peak_pred[1])**2)**0.5

        weights.append(error_peaks.numpy())

        for i in range(1, self.num_harmonics_optimizer + 1):
            fragment_true = fft_true[
                :, self.freq_bounds[i] : self.freq_bounds[i + 1], :
            ]
            fragment_pred = fft_pred[
                :, self.freq_bounds[i] : self.freq_bounds[i + 1], :
            ]
            peak_true = (self.find_peak_x(fragment_true), self.find_peak_y(fragment_true))
            peak_pred = (self.find_peak_x(fragment_pred), self.find_peak_y(fragment_pred))

            error_peaks = ((peak_true[0]-peak_pred[0])**2 + (peak_true[1]-peak_pred[1])**2)**0.5

            weights.append(error_peaks.numpy())
        
        ### Normalize weights so that the sum equals 1
        self.weights[1:] = weights
        sum_list = sum(self.weights)
        self.weights = [element / sum_list for element in self.weights]
        
    def custom_loss_function(self, y_true, y_pred):

        # tf.print("\nCUSTOM_LOSS_FUNCTION")

        mse = tf.keras.losses.MeanSquaredError()

        ### Remove the mean from the signal
        y_true -= tf.math.reduce_mean(y_true)
        y_pred -= tf.math.reduce_mean(y_pred)

        y_true_complex = tf.cast(y_true, tf.complex64)
        y_pred_complex = tf.cast(y_pred, tf.complex64)

        fft_true = tf.signal.fft2d(y_true_complex)
        fft_pred = tf.signal.fft2d(y_pred_complex)
        fft_true = 2.0 / self.N * tf.math.abs(fft_true[:, 0 : self.N // 2, :])
        fft_pred = 2.0 / self.N * tf.math.abs(fft_pred[:, 0 : self.N // 2, :])

        # y_true *= self.compute_multiplication_factor_tf(y_true)
        # y_pred *= self.compute_multiplication_factor_tf(y_pred)

        loss_comp = list()
        loss_wgt = list()
        loss_abs = list()
        
        peaks_true = list()
        peaks_pred = list()

        loss_abs.append(mse(y_true, y_pred))
        loss_wgt.append(self.weights[0] * mse(y_true, y_pred))
        loss_comp.append(
            (
                (self.weights[0]*(mse(y_true, y_pred) - self.target_obj_function))
                / (self.worst_f_value_time - self.target_obj_function)
            )
            ** 2
        )

        # """
        if self.num_harmonics_optimizer > 0:

            # self.dynamically_adjust_weights(fft_true,fft_pred)

            fragment_true = fft_true[:, self.freq_bounds[0] : self.freq_bounds[1], :]
            fragment_pred = fft_pred[:, self.freq_bounds[0] : self.freq_bounds[1], :]

            # fragment_true *= self.compute_multiplication_factor_tf(fragment_true)
            # fragment_pred *= self.compute_multiplication_factor_tf(fragment_pred)

            loss_abs.append(mse(fragment_true, fragment_pred))
            loss_wgt.append(self.weights[1] * mse(fragment_true, fragment_pred))
            loss_comp.append(
                (
                    (
                        self.weights[1]
                        * (mse(fragment_true, fragment_pred) - self.target_obj_function)
                    )
                    / (self.worst_f_values_freq[0] - self.target_obj_function)
                )
                ** 2
            )

            for i in range(1, self.num_harmonics_optimizer + 1):
                fragment_true = fft_true[
                    :, self.freq_bounds[i] : self.freq_bounds[i + 1], :
                ]
                fragment_pred = fft_pred[
                    :, self.freq_bounds[i] : self.freq_bounds[i + 1], :
                ]

                # fragment_true *= self.compute_multiplication_factor_tf(fragment_true)
                # fragment_pred *= self.compute_multiplication_factor_tf(fragment_pred)

                loss_abs.append(mse(fragment_true, fragment_pred))
                loss_wgt.append(self.weights[i + 1] * mse(fragment_true, fragment_pred))
                loss_comp.append(
                    (
                        (
                            self.weights[i + 1] * mse(fragment_true, fragment_pred)
                            - self.target_obj_function
                        )
                        / (self.worst_f_values_freq[i] - self.target_obj_function)
                    )
                    ** 2
                )

        loss = tf.math.reduce_sum(loss_comp)
        ### Apply the squared root
        loss = loss**0.5

        return loss

### Out of the class
def cosines_activation_funtion(x):
    return (1 / 2) * (K.cos(x + 3 * math.pi / 2) + 1)