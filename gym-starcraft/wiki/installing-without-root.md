```
- mkdir ~/Public
- mkdir ~/apps
- cd ~/Public
- wget https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz
- tar xf zeromq-4.1.4.tar.gz
- wget https://download.libsodium.org/libsodium/releases/libsodium-1.0.14.tar.gz
- tar xf libsodium-1.0.14.tar.gz
- cd libsodium-1.0.13
- ./configure --prefix=$HOME/apps
- make && make check
- make install
- cd .. && cd zeromq-4.1.4
- PKG_CONFIG_PATH=$HOME/apps/lib/pkgconfig ./configure --prefix $HOME/apps
- make
- make install
- export CFLAGS="-I$HOME/apps/include"
- export LDFLAGS="-L$HOME/apps/lib"
- export LD_LIBRARY_PATH="$HOME/apps/usr/local/lib:$HOME/apps/lib:$LD_LIBRARY_PATH"
- cd .. && git clone https://github.com/facebook/zstd
- cd zstd && DESTDIR=$HOME/apps make -j4 install && cd ..
- git clone https://github.com/openbw/openbw
- git clone https://github.com/openbw/bwapi
- cd bwapi
- mkdir build && cd build
The command below might need CC=gcc CXX=g++ to specify particular versions of gcc.
- cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENBW_DIR=../../openbw -DOPENBW_ENABLE_UI=1 -DCMAKE_INSTALL_PREFIX=~/Public/bwapi
If the above doesn't work we need sdl2 or install without -DOPENBW_ENABLE_UI option
	- wget https://libsdl.org/release/SDL2-2.0.8.tar.gz && tar SDL2-2.0.8 && cd SDL2-2.0.8
	- mkdir build && cd build && ../configure --prefix $HOME/apps
	- make && make install
	- Add SDL2 to path
	- cd ../bwapi/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENBW_DIR=../../openbw -DOPENBW_ENABLE_UI=1 -			DCMAKE_INSTALL_PREFIX=~/Public/bwapi
- make install
- cd ../.. && git clone https://github.com/TorchCraft/TorchCraft && cd TorchCraft && git fetch origin develop:develop
- git submodule update --init --recursive
- cd BWEnv && mkdir -p build && cd build

- You will need g++ version > 5.0.0 for easy installation
- CC=gcc CXX=g++ CXXFLAGS="-I$HOME/apps/usr/local/include -I$HOME/apps/include" cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo -DBWAPI_DIR=../../bwapi/ && make -j
- Finally python installation
- cd ../.. && pip install pybind11
- LDFLAGS="-L$HOME/apps/usr/local/lib -L$HOME/apps/lib" CFLAGS="-I$HOME/apps/usr/local/include -I$HOME/apps/include" pip install -e .
- Add following lines to .zshrc or .bashrc

export CFLAGS="-I$HOME/apps/usr/local/include -I$HOME/apps/include"
export LDFLAGS="-L$HOME/apps/usr/local/lib -L$HOME/apps/lib"
export LD_LIBRARY_PATH="$HOME/apps/usr/local/lib:$HOME/apps/lib:$LD_LIBRARY_PATH"

- Always use the g++ you used to compile while running
- Copy .mpq files to TorchCraft folder and you are ready to run.
```
