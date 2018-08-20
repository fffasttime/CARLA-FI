import os
import glob
import h5py
import traceback
import sys
import math
import copy
import json

import numpy as np

from torch.utils.data import Dataset
import torch

from logger import coil_logger

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from utils.general import sort_nicely



class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None):  # The transformation object.
        """
        Function to encapsulate the dataset

        Arguments:
            root_dir (string): Directory with all the hdfiles from the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.sensor_data, self.measurements, self.meta_data = self.pre_load_hdf5_files(root_dir)
        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        # This is seems to be the entire dataset size

        return self.measurements.shape[1]

    def __getitem__(self, used_ids):
        """
        Function to get the items from a dataset

        Arguments
            us
        """
        # We test here directly and include the other images here.
        batch_sensors = {}

        # Number of positions

        try:
            number_of_position = len(used_ids)
        except:
            number_of_position = 1
            used_ids = [used_ids]

        float_data = self.measurements[:, used_ids]

        # Initialization of the numpy arrays
        for sensor_name, sensor_size in g_conf.SENSORS.items():
            sensor_data = np.zeros(
                (number_of_position, sensor_size[0], sensor_size[1],
                 sensor_size[2] * g_conf.NUMBER_FRAMES_FUSION),
                dtype='float32'
            )


            batch_sensors.update({sensor_name: sensor_data})

        for sensor_name, sensor_size in g_conf.SENSORS.items():
            count = 0
            for chosen_key in used_ids:

                for i in range(g_conf.NUMBER_FRAMES_FUSION):
                    chosen_key = chosen_key + i * 3


                    """
                    for es, ee, x in self.sensor_data[count]:

                        if chosen_key >= es and chosen_key < ee:


                            pos_inside = chosen_key - es
                            sensor_image = np.array(x[pos_inside, :, :, :])
                    """


                    """ We found the part of the data to open """

                    pos_inside = chosen_key - (chosen_key // 200)*200
                    # TODO: converting to images. The two goes out.
                    sensor_image = self.sensor_data[count][chosen_key // 200][2][pos_inside]


                    if self.transform is not None:
                        sensor_image = self.transform(self.batch_read_number, sensor_image)
                    else:

                        sensor_image = np.swapaxes(sensor_image, 0, 2)
                        sensor_image = np.swapaxes(sensor_image, 1, 2)

                    # Do not forget the final normalization step
                    batch_sensors[sensor_name][count, (i * 3):((i + 1) * 3), :, :
                    ] = sensor_image/255.0

                    del sensor_image



                count += 1

        #TODO: if experiments change name there should be an error

        if g_conf.AUGMENT_LATERAL_STEERINGS > 0:

            camera_angle = float_data[np.where(self.meta_data[:, 0] == b'angle'), :][0][0]
            speed = float_data[np.where(self.meta_data[:, 0] == b'speed_module'), :][0][0]
            steer = float_data[np.where(self.meta_data[:, 0] == b'steer'), :][0][0]

            float_data[np.where(self.meta_data[:, 0] == b'steer'), :] =\
                self.augment_steering(camera_angle, copy.copy(steer), speed)


            #print ( 'camera angle', camera_angle,
            #        'new_steer' ,float_data[np.where(self.meta_data[:, 0] == b'steer'), :],
            #       'old_steer', steer)

        float_data[np.where(self.meta_data[:, 0] == b'speed_module'), :] /= g_conf.SPEED_FACTOR




        self.batch_read_number += 1
        # TODO: IMPORTANT !!!
        # TODO: ADD GROUND TRUTH CONTROL IN SOME META CONFIGURATION FOR THE DATASET
        # TODO: SO if the data read and manipulate is outside some range, it should report error
        return batch_sensors, float_data

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras.
        Args:
            camera_angle_batch:
            steer_batch:
            speed_batch:

        Returns:
            the augmented steering

        """

        time_use = 1.0
        car_length = 6.0
        old_steer = steer
        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
        math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))


        #print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer

    def augment_measurement(self, measurements, angle):
        """

            Augment the steering of a measurement dict


        """

        new_steer = self.augment_steering(angle, measurements['steer'],
                                          measurements['playerMeasurements']['forwardSpeed'])

        measurements['steer'] = new_steer
        return measurements['steer']

    def pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now

        args
            the path for the dataset


        returns
         sensor data names: it is a vector with n dimensions being one for each sensor modality
         for instance, rgb only dataset will have a single vector with all the image names.
         float_data: all the wanted float data is loaded inside a vector, that is a vector
         of dictionaries.

        """

        episodes_list = glob.glob(os.path.join(path, 'episode_*'))

        sensor_data_names = []
        float_dicts = []


        for episode in episodes_list:
            print ('Episode ', episode)

            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)

            for measurement in measurements_list:
                data_point_number = measurement.split('_')[-1].split('.')[0]

                #TODO the dataset camera name can be a parameter
                with open(measurement) as f:
                    measurement_data = json.load(f)
                # We extract the interesting subset from the measurement dict
                float_dicts.append(
                    {'steer': measurement_data['steer'],
                     'throttle':  measurement_data['throttle'],
                     'brake': measurement_data['brake'],
                     'speed_module': measurement_data['playerMeasurements']['forwardSpeed'],
                     'directions': measurement_data['directions']}
                )

                rgb = 'CentralRGB_' + data_point_number + '.jpg'
                sensor_data_names.append(os.path.join(episode, rgb))

                # We do measurements for the left side camera
                # #TOdo the angle does not need to be hardcoded
                measurement_left = self.augment_measurement(measurement_data, -30.0)

                # We extract the interesting subset from the measurement dict
                float_dicts.append(
                    {'steer': measurement_left['steer'],
                     'throttle':  measurement_left['throttle'],
                     'brake': measurement_left['brake'],
                     'speed_module': measurement_left['playerMeasurements']['forwardSpeed'],
                     'directions': measurement_left['directions']}
                )
                rgb = 'LeftRGB_' + data_point_number + '.jpg'
                sensor_data_names.append(os.path.join(episode, rgb))

                # We do measurements augmentation for the right side cameras

                measurement_right = self.augment_measurement(measurement_data, 30.0)
                float_dicts.append(
                    {'steer': measurement_right['steer'],
                     'throttle':  measurement_right['throttle'],
                     'brake': measurement_right['brake'],
                     'speed_module': measurement_right['playerMeasurements']['forwardSpeed'],
                     'directions': measurement_right['directions']}
                )
                rgb = 'RightRGB_' + data_point_number + '.jpg'
                sensor_data_names.append(os.path.join(episode, rgb))




        return sensor_data_names, float_dicts




    # file_names, image_dataset_names, dataset_names
    def pre_load_hdf5_files(self, path_for_files):
        """
        Function to load all hdfiles from a certain folder
        TODO: Add partially loading of the data
        Returns
            TODO: IMPROVE
            A list with the read sensor data ( h5py)

            All the measurement data

        """

        # Take the names of all measurements from the dataset
        meas_names = list(g_conf.MEASUREMENTS.keys())
        # take the names for all sensors
        sensors_names = list(g_conf.SENSORS.keys())

        # From the determined path take all the possible file names.
        # TODO: Add more flexibility for the file base names ??
        folder_file_names = [os.path.join(path_for_files, f)
                             for f in glob.glob1(path_for_files, "data_*.h5")]

        folder_file_names = sorted(folder_file_names)


        # THIS WILL CHECK IF THIS DATASET IS VALID
        """
        while True:
            try:
                if not is_hdf5_prepared(folder_file_names[0]):
                    raise ValueError("The showed dataset is not prepared for training")
                break
            except OSError:
                import traceback
                time.sleep(0.5)
                traceback.print_exc()
                continue
        """


        # Concatenate all the sensor names and measurements names
        # TODO: This structure is very ugly.
        meas_data_cat = [list([]) for _ in range(len(meas_names))]
        sensors_data_cat = [list([]) for _ in range(len(sensors_names))]

        # We open one dataset to get the metadata for targets
        # that is important to be able to reference variables in a more legible way
        dataset = h5py.File(folder_file_names[0], "r")
        metadata_targets = np.array(dataset['metadata_' + meas_names[0]])
        dataset.close()

        # Forcing the metadata to be bytes
        if not isinstance(metadata_targets[0][0], bytes):
            metadata_targets = np.array(
                [[some_meta_data[0].encode('utf-8'), some_meta_data[1].encode('utf-8')]
                 for some_meta_data in metadata_targets])

        lastidx = 0
        count = 0
        coil_logger.add_message('Loading', {'FilesLoaded': folder_file_names,
                                            'NumberOfImages': len(folder_file_names)})

        for file_name in folder_file_names:
            try:
                dataset = h5py.File(file_name, "r")

                for i in range(len(sensors_names)):
                    x = dataset[sensors_names[i]]
                    old_shape = x.shape[0]
                    #  Concatenate all the datasets for a given sensor.
                    sensors_data_cat[i].append((lastidx, lastidx + x.shape[0], x))

                for i in range(len(meas_names)):
                    dset_to_append = dataset[meas_names[i]]
                    meas_data_cat[i].append(dset_to_append[:])

                lastidx += old_shape
                dataset.flush()
                count += 1

            except IOError:

                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                print("failed to open", file_name)


        # For the number of datasets names that are going to be used for measurements cat all.
        for i in range(len(meas_names)):
            #print (meas_data_cat[i])
            meas_data_cat[i] = np.concatenate(meas_data_cat[i], axis=0)
            meas_data_cat[i] = meas_data_cat[i].transpose((1, 0))

        return sensors_data_cat, meas_data_cat[0], metadata_targets

    # TODO: MAKE AN "EXTRACT" method used by both of the functions above.

    # TODO: Turn into a static property

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]

    def extract_targets(self, float_data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(float_data[:, np.where(self.meta_data[:, 0] == target_name.encode())
                                             [0][0], :])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, float_data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(float_data[:, np.where(self.meta_data[:, 0] == input_name.encode())
                                            [0][0], :])

        return torch.cat(inputs_vec, 1)