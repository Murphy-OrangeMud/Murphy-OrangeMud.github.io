---
title: "How to Deploy Mininet on M1 Mac"
date: 2022-10-04T23:08:48-04:00
draft: false
tags: ["M1 Mac"]
---

On M1 Mac, we can't walkthrough Mininet's official tutorial because it used Virtualbox, which is not supported on M1 Mac ~~and have no recent announcement on this work~~. So we'll try Qemu.

First download Mininet Image from [this link](https://github.com/mininet/mininet/releases)

And we can directly use the following command:
``` bash
qemu-system-x86_64 \
  -m 8192 \
  mininet-vm-x86_64.vmdk \
  -net nic,model=virtio \
  -net user,net=192.168.18.0/24,hostfwd=tcp::8022-:22
```

Some explanation:
`-m 8192` means to set the memory to 8192MiB
`.vmdk` is the image file of Mininet we have just downloaded
`-net mic` demonstrate that the network is connected to a network interface controller
`model=virtio` is a deprecated feature, meaning that the I/O are virtual
`-net user,net=192.168.18.0/24,hostfwd=tcp::8022-:22` means that the network architecture is user network and the visible network segment for guest system is 192.168.18.0/24. The port forwarding rule is 8022 port on host machine to 22 port on guest machine

and to use wireshark, we ssh into the virtual machine from the host terminal:
```
ssh mininet@localhost -p 8022
```

### Troublshooting
I met with several bugs during this process.

##### 1st BUG:
```
$DISPLAY not set
```

##### 2nd BUG:
```
qt.qpa.xcb: could not connect to display
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, xcb.

Aborted
```

I executed these according to the documentation:
```
$ dpkg -l | grep libdouble-conversion  # to see which version you have
$ sudo apt remove libdouble-conversion3  # be sure to specify the right version
$ sudo apt autoremove
$ sudo apt install wireshark
```

Add -X after ssh:
```
X11 connection rejected because of wrong authentication.
```

And these are the final solutions:

**Step 0**
```
sudo apt-get update
sudo apt-get upgrade
```

**Step 1**
```
qt.qpa.xcb: could not connect to display localhost:10.0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, xcb.

Aborted
```

There are displays!

**Step 2**
We can see that:
```
/usr/bin/xauth:  file /home/mininet/.Xauthority does not exist
```
But that's OK because it will be constructed automatically. But we shall check if it does exist, or we should construct manually.

**Step 3**
Both on host and on guest we run:
``` bash
sudo vim /etc/ssh/sshd_config
```

Change the option `X11Forwarding` to yes.

And on host machine we uncomment (maybe unnecessary):
```
X11DisplayOffset 10
X11UseLocalhost yes
```

**Step 4**
We run:
``` bash
export XAUTHORITY=$HOME/.Xauthority
```
