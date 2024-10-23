############
ts_aos_utils
############

Requirements
------------

This package requires some extra repositories to be cloned locally so it can grab some constants and some look-up tables. Here is a list of which repos are required to run this notebook:

[ts_config_mttcs](https://github.com/lsst-ts/ts_config_mttcs)
[ts_criopy](https://github.com/lsst-ts/ts_criopy)
[ts_m2com](https://github.com/lsst-ts/ts_m2com)
[ts_tcpip](https://github.com/lsst-ts/ts_tcpip)
[ts_utils](https://github.com/lsst-ts/ts_utils)
[ts_xml](https://github.com/lsst-ts/ts_xml)

Since every user has a different setup, the paths might be slightly different. It is recommended to have all the repositories cloned under $HOME/notebooks. You might end up with many repositories and adding an extra folder with the name of the organization they belong might help to find them on GitHub later. For example, if you are working under notebooks_vandv environment, you should clone and install packages in the folder $HOME/notebooks/lsst-ts. The paths below consider this directory structure but, of course, you are free to organize your folders as you please.

To have the required repositories available, open a terminal and run the following commands:

```

    git clone https://github.com/lsst-ts/ts_aos_utils $HOME/notebooks/lsst-ts/ts_aos_utils
    git clone https://github.com/lsst-ts/ts_criopy $HOME/notebooks/lsst-ts/ts_criopy
    git clone (https://github.com/lsst-ts/ts_m2com) $HOME/notebooks/lsst-ts/ts_m2com
    git clone https://github.com/lsst-ts/ts_config_mttcs $HOME/notebooks/lsst-ts/ts_config_mttcs
    git clone https://github.com/lsst-ts/ts_utils $HOME/notebooks/lsst-ts/ts_utils
    git clone https://github.com/lsst-ts/ts_xml $HOME/notebooks/lsst-ts/ts_xml
    git clone https://github.com/lsst-ts/ts_tcpip $HOME/notebooks/lsst-ts/lsst/ts/ts_tcpip    
```

If you use a different path for these repositories, make sure that you pass this path when running the associated functions.

And add these lines to your $HOME/notebooks/.user_setups file:

```

    setup -j ts_aos_utils -r $HOME/notebooks/lsst-ts/ts_aos_utils
    setup -j ts_criopy -r $HOME/notebooks/lsst-ts/ts_criopy
    setup -j ts_m2com -r $HOME/notebooks/lsst-ts/ts_m2com
    setup -j ts_config_mttc -r $HOME/notebooks/lsst-ts/ts_config_mttcs
    setup -j ts_utils -r $HOME/notebooks/lsst-ts/ts_utils
    setup -j ts_xml -r $HOME/notebooks/lsst-ts/ts_xml
    setup -j ts_tcpip -r $HOME/notebooks/lsst-ts/ts_tcpip
```    


