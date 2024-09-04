import numpy as np
import h5py


class DatasetTimes:
    def __init__(self, parent_ims, mode, time=None):
        """Creates DataSetTimes class object and .ims Group class object with
        appropriate attributes.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
            time (str): Optional string containing the time file was created at.
        """
        if mode == 'x':
            self.imsgroup = parent_ims.create_group("DataSetTimes")
            self.time_datatype = np.dtype([('ID', np.uint64),
                                           ('Birth', np.uint64),
                                           ('Death', np.uint64),
                                           ('IDTimeBegin', np.uint64),
                                           ])

            self.time_arr = np.array([(0, 0, 1000000000, 0)],
                                     dtype=self.time_datatype)
            self.time_ims = self.imsgroup.create_dataset("Time",
                                                         data=self.time_arr)
            self.timebegin_datatype = np.dtype([('ID', np.uint64),
                                                ('ObjectTimeBegin',
                                                 h5py.special_dtype(vlen=str))
                                                ])
            self.timebegin_arr = np.array([(0, time)],
                                          dtype=self.timebegin_datatype)
            self.timebegin_ims = self.imsgroup.create_dataset("TimeBegin",
                                                              data=self.
                                                              timebegin_arr)
        if mode == 'r+':
            self.imsgroup = parent_ims["DataSetTimes"]
            self.time_ims = self.imsgroup["Time"]
            self.timebegin_ims = self.imsgroup["TimeBegin"]


class Scene:
    def __init__(self, parent_ims, mode):
        """Creates Scene and Scene8 class objects and .ims Group class objects
        with appropriate attributes.

        Args:
            parent_ims (obj): A parent group object from h5py module.
            mode (str): r+: states there is an existing ims file that the object
                of this class should read or x: creates a new imaris file.
        """
        if mode == 'x':
            self.scene = {"Scene": parent_ims.create_group("Scene"),
                          "Scene8": parent_ims.create_group("Scene8")
                          }
            for scene in self.scene.values():
                scene.create_group("Content")
        if mode == 'r+':
            self.scene = {"Scene": parent_ims["Scene"],
                          "Scene8": parent_ims["Scene8"]
                          }
