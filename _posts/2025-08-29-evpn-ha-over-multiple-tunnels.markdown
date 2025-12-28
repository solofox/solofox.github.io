---
layout: post
title:  "SDN&网络：EVPN高可用实践"
date:   2025-08-29 22:56:00 +0800
categories: networking SDN EVPN BGP
---
EVPN的高可用，在RFC中主推多归（Multi Home，MH），里面引入了ES/ESI/DF等概念，复杂度高。既然EVPN是基于BGP的，我们能不能在不引入MH，支持EVPN的高可靠呢？答案是可以的。

# 需求
如图，每台服务器分别连接到两个网络中，eth1连接到网络1，eth2连接到网络2。这些连接可能不稳定而产生中断（它们是基于internet的隧道），网络1和网络2能够形成主备关系。对于业务而言，希望服务地址跟底层的连接状况无关，因此单独分配一个业务网口，业务使用业务IP对外通信。

![network topology and requirements](/assets/2025-08-29/requirements.png "network topology and requirements")

我们想象中，从VTEP的基础出发，右边的server-r通过EVPN宣告第一组主路由：
> 1. 192.168.0.87对应的mac是00:01:02:a8:00:57
> 2. 00:01:02:a8:00:57在192.168.1.87这个VTEP上

有了这两个信息之后，左边的server-l下发IP邻居表和bridge FDB，从左边到右边的VXLAN数据通路就打通了。这两个信息加起来，正是evpn type2路由。

为了实现高可用，server-r还需要通过EVPN宣告第二组备份路由：
> 1. 192.168.0.87对应的mac是00:01:02:a8:00:57，这个信息跟第一组是完全一样的
> 2. 00:01:02:a8:00:57在192.168.2.87这个VTEP上

当网络1的连接中断后，server-l撤销主路由生成的IP邻居表和bridge FDB，根据第二组type2 路由生成新的表项，实现了VXLAN网络的高可用。

let's do it now.

# FRR方案

FRR不多介绍。整个方案在server上创建VTEP口（不要指定源地址），创建业务loopback口，让bgpd感知到这个loopback口的IP、MAC、VXLAN ID，然后在宣告路由时选择本地接口地址当做nexthop（VXLAN VTEP地址）即可。

![FRR solution for VXLAN high-availability](/assets/2025-08-29/frr-solution.png "FRR solution for VXLAN high-availability")

为了宣告路由，需要在每个网络中引入路由反射器RR：RR-net1和RR-net2。作为POC，这里并没有考虑RR的高可用。RR的选择可以和服务器解构，可以不用FRR，选择GoBGP、bird，甚至支持EVPN的路由器都可以。

FRR方案的限制：
> 1. 限于FRR，无法把网络1和网络2划分为两个ASN域（比较复杂，需要用路由重分发），因此两个RR都是在65001中。
> 2. 无法控制主备，只能依靠FRR hardcode的规则，你需要给主链路配置更小的router id。

## RR的配置

下面给出了RR-net1的frr.conf，RR-net2的配置参考修改相关的IP等。

```txt [frr.conf for RR-net1]
router bgp 65001
 bgp router-id 192.168.1.2
 bgp log-neighbor-changes
 no bgp default ipv4-unicast
 bgp cluster-id 192.168.1.2
 neighbor VTEPs peer-group
 neighbor VTEPs remote-as 65001
 neighbor VTEPs capability extend-nexthop
 ! clients
 bgp listen range 192.168.1.0/24 peer-group VTEPs
 ! 
 address-family l2vpn evpn
  neighbor VTEPs activate
  neighbor VTEPs route-reflector-client
 exit-address-family
exit
```


## 服务器的配置

下面给出了server-l的相关配置，server-r的配置参考修改相关的IP等。VXLAN id取100。

### 网络接口配置

 ```bash [vxlan接口，网桥初始化脚本]
 set -e 
 ip link add vxlan0 type vxlan 100 dstport 4789 nolearning
 ip link set vxlan up
 ip link add br-vxlan type bridge
 ip link set vxlan0 master br-vxlan
 ip addr add 192.168.0.65/24 dev br-vxlan
 ip link set br-vxlan up
 ```

### 控制面配置
这里的控制面就是frr的bgp配置：

```text [frr.conf for server-l]
! 禁用type3路由。这个VXLAN网络中无未知/隐藏的服务器，无需BUM处理。
route-map NO_TYPE3 deny 10
 match evpn route-type 3
route-map NO_TYPE3 permit 20
! 主备，实际上不work
route-map PRIMARY-LINK permit 10
 set local-preference 200
route-map BACK-LINK permit 10
 set local-prefernce 100
router bgp 65001
 bgp router-id 192.168.1.65
 ! bgp listen disable
 no bgp default ipv4-unicast
 neighbor RR peer-group
 neighbor RR remote-as 65001
 neighbor RR capability extend-nexthop
 neighbor RR timers 3 9
 ! neighbors
 neighbor 192.168.1.2 peer-group RR
 neighbor 192.168.2.2 peer-group RR
 address-family l2vpn evpn
  neighbor RR route-map NO_TYPE3 out
  neighbor 192.168.1.2 route-map PRIMARY-LINK in
  neighbor 192.168.2.2 route-map BACKUP-LINK in
  neighbor RR next-hop-self
  neighbor RR activate
  advertise-all-vni
  advertise-svi-ip
 exit-address-family
exit
 ```

## 一键部署 
[这里](https://github.com/solofox/evpn-ha/tree/main/frr)是一键部署terraform脚本，天下武功，为快不破！

## 验证

### 正常情况

```bash
# 看一下系统网络的配置
root@server-l:~# ifconfig | egrep 'mtu|inet|eth' 
br-vxlan: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.0.65  netmask 255.255.255.0  broadcast 0.0.0.0
        inet6 fe80::c896:f2ff:fe80:73e6  prefixlen 64  scopeid 0x20<link>
        ether ca:96:f2:80:73:e6  txqueuelen 1000  (Ethernet)
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.1.65  netmask 255.255.255.0  broadcast 192.168.1.255
        inet6 fe80::216:3eff:fe36:3738  prefixlen 64  scopeid 0x20<link>
        ether 00:16:3e:36:37:38  txqueuelen 1000  (Ethernet)
eth1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.2.65  netmask 255.255.255.0  broadcast 192.168.2.255
        inet6 fe80::216:3eff:fe36:386d  prefixlen 64  scopeid 0x20<link>
        ether 00:16:3e:36:38:6d  txqueuelen 1000  (Ethernet)
lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
vxlan0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet6 fe80::fcb3:86ff:fe6d:17d8  prefixlen 64  scopeid 0x20<link>
        ether fe:b3:86:6d:17:d8  txqueuelen 1000  (Ethernet)

# ping server-r的业务口
root@server-l:~# ping -c 1 192.168.0.87
PING 192.168.0.87 (192.168.0.87) 56(84) bytes of data.
64 bytes from 192.168.0.87: icmp_seq=1 ttl=64 time=0.507 ms

--- 192.168.0.87 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.507/0.507/0.507/0.000 ms

# 抓包确认，只有两个包，icmp echo request和icmp echo reply。注意没有ARP报文。
root@server-l:~# tcpdump -lenv -i any udp port 4789
tcpdump: data link type LINUX_SLL2
tcpdump: listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
14:45:30.398404 eth0  Out ifindex 2 00:16:3e:36:37:38 ethertype IPv4 (0x0800), length 154: (tos 0x0, ttl 64, id 1338, offset 0, flags [none], proto UDP (17), length 134)
    192.168.1.65.38087 > 192.168.1.87.4789: VXLAN, flags [I] (0x08), vni 100
ca:96:f2:80:73:e6 > ae:bf:39:cc:6b:ed, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 33244, offset 0, flags [DF], proto ICMP (1), length 84)
    192.168.0.65 > 192.168.0.87: ICMP echo request, id 5023, seq 1, length 64
14:45:30.398818 eth0  In  ifindex 2 ee:ff:ff:ff:ff:ff ethertype IPv4 (0x0800), length 154: (tos 0x0, ttl 64, id 35750, offset 0, flags [none], proto UDP (17), length 134)
    192.168.1.87.41404 > 192.168.1.65.4789: VXLAN, flags [I] (0x08), vni 100
ae:bf:39:cc:6b:ed > ca:96:f2:80:73:e6, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 58674, offset 0, flags [none], proto ICMP (1), length 84)
    192.168.0.87 > 192.168.0.65: ICMP echo reply, id 5023, seq 1, length 64

# ip neigh，确实有
root@server-l:~# ip neigh get 192.168.0.87 dev br-vxlan
192.168.0.87 dev br-vxlan lladdr ae:bf:39:cc:6b:ed extern_learn NOARP proto zebra 
# bridge fdb，下一条VTEP IP是192.168.1.87
root@server-l:~# bridge fdb  | grep 'ae:bf:39:cc:6b:ed'
ae:bf:39:cc:6b:ed dev vxlan0 vlan 1 extern_learn master br-vxlan 
ae:bf:39:cc:6b:ed dev vxlan0 extern_learn master br-vxlan 
ae:bf:39:cc:6b:ed dev vxlan0 dst 192.168.1.87 self extern_learn 

# 看看bgp中学到的路由
root@server-l:~# vtysh -c 'show bgp l2vpn evpn'
BGP table version is 3, local router ID is 192.168.1.65
Status codes: s suppressed, d damped, h history, * valid, > best, i - internal
Origin codes: i - IGP, e - EGP, ? - incomplete
EVPN type-1 prefix: [1]:[EthTag]:[ESI]:[IPlen]:[VTEP-IP]:[Frag-id]
EVPN type-2 prefix: [2]:[EthTag]:[MAClen]:[MAC]:[IPlen]:[IP]
EVPN type-3 prefix: [3]:[EthTag]:[IPlen]:[OrigIP]
EVPN type-4 prefix: [4]:[ESI]:[IPlen]:[OrigIP]
EVPN type-5 prefix: [5]:[EthTag]:[IPlen]:[IP]

   Network          Next Hop            Metric LocPrf Weight Path
Route Distinguisher: 192.168.1.65:2
*> [2]:[0]:[48]:[ca:96:f2:80:73:e6]:[32]:[192.168.0.65]
                    192.168.1.65                       32768 i
                    ET:8 RT:65001:100
*> [2]:[0]:[48]:[ca:96:f2:80:73:e6]:[128]:[fe80::c896:f2ff:fe80:73e6]
                    192.168.1.65                       32768 i
                    ET:8 RT:65001:100
*> [3]:[0]:[32]:[192.168.1.65]
                    192.168.1.65                       32768 i
                    ET:8 RT:65001:100
Route Distinguisher: 192.168.1.87:2
* i[2]:[0]:[48]:[ae:bf:39:cc:6b:ed]:[32]:[192.168.0.87]
                    192.168.2.87             0    100      0 i
                    RT:65001:100 ET:8
*>i                 192.168.1.87             0    200      0 i
                    RT:65001:100 ET:8
* i[2]:[0]:[48]:[ae:bf:39:cc:6b:ed]:[128]:[fe80::fcf0:17ff:fe81:797f]
                    192.168.2.87             0    100      0 i
                    RT:65001:100 ET:8
*>i                 192.168.1.87             0    200      0 i
                    RT:65001:100 ET:8

Displayed 5 out of 7 total prefixes
```

### 切换
增加iptables模拟一下故障，再测试一下
```bash
# 故障注入
root@server-l:~# iptables -A INPUT -p tcp -s 192.168.1.2 --sport 179 -j DROP
root@server-l:~# iptables -A INPUT -p udp -s 192.168.1.0/24 --dport 4789 -j DROP

# ping验证，还是通的
root@server-l:~# ping -c1 192.168.0.87
PING 192.168.0.87 (192.168.0.87) 56(84) bytes of data.
64 bytes from 192.168.0.87: icmp_seq=1 ttl=64 time=0.114 ms

--- 192.168.0.87 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.114/0.114/0.114/0.000 ms

# 抓包确认下
root@server-l:~# tcpdump -lenv -i any udp port 4789
tcpdump: data link type LINUX_SLL2
tcpdump: listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
14:59:06.362474 eth1  Out ifindex 3 00:16:3e:36:38:6d ethertype IPv4 (0x0800), length 154: (tos 0x0, ttl 64, id 59948, offset 0, flags [none], proto UDP (17), length 134)
    192.168.2.65.38087 > 192.168.2.87.4789: VXLAN, flags [I] (0x08), vni 100
ca:96:f2:80:73:e6 > ae:bf:39:cc:6b:ed, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 18261, offset 0, flags [DF], proto ICMP (1), length 84)
    192.168.0.65 > 192.168.0.87: ICMP echo request, id 5082, seq 1, length 64
14:59:06.362566 eth1  In  ifindex 3 ee:ff:ff:ff:ff:ff ethertype IPv4 (0x0800), length 154: (tos 0x0, ttl 64, id 24233, offset 0, flags [none], proto UDP (17), length 134)
    192.168.2.87.41404 > 192.168.2.65.4789: VXLAN, flags [I] (0x08), vni 100
ae:bf:39:cc:6b:ed > ca:96:f2:80:73:e6, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 12755, offset 0, flags [none], proto ICMP (1), length 84)
    192.168.0.87 > 192.168.0.65: ICMP echo reply, id 5082, seq 1, length 64

# ip neigh没什么变化
root@server-l:~# ip neigh get 192.168.0.87 dev br-vxlan
192.168.0.87 dev br-vxlan lladdr ae:bf:39:cc:6b:ed extern_learn NOARP proto zebra 
# bridge fdb 显示的VTEP IP确切换了
root@server-l:~# bridge fdb  | grep 'ae:bf:39:cc:6b:ed'
ae:bf:39:cc:6b:ed dev vxlan0 vlan 1 extern_learn master br-vxlan 
ae:bf:39:cc:6b:ed dev vxlan0 extern_learn master br-vxlan 
ae:bf:39:cc:6b:ed dev vxlan0 dst 192.168.2.87 self extern_learn 
```

# GoBGP方案
上面提到frr有两个限制:
> 1. 限于FRR，无法把网络1和网络2划分为两个ASN域（比较复杂，需要用路由重分发），因此两个RR都是在65001中。
> 2. 无法控制主备，只能依靠FRR hardcode的规则，你需要给主链路配置更小的router id。

用GoBGP的方案运行两个GoBGP来解决这两个限制，但是GoBGP需要自己写一个vxlan.py来完成配置下发到kernel的任务：
> 写一个python程序，它用gprc从gobgpd监听中的evpn的type2路由变化，进行优选，并通过netlink添加和删除neigh和bridge fwd表。
> 程序启动时，需要把vxlan0的ip和mac，加上网卡IP坐下一下跳，宣告出去
没有grpc和Linux网络子系统netlink编程的朋友们会比较费劲，好在我们现在有AI。

![GoBGP solution for VXLAN high-availability](/assets/2025-08-29/gobgp-solution.png "GoBGP solution for VXLAN high-availability")

## RR的配置
RR不是这套方案的关键，RR还是继续沿用FRR方案中的配置，除了把RR-net2的asn改为65002。

## 服务器的配置

### 网络接口配置

```bash
ip link add br-vxlan type bridge
ip addr add 192.168.0.65/24 dev br-vxlan
ip link set br-vxlan up

# 创建VTEP接口
ip link add vxlan0 type vxlan id ${var.vni} dstport 4789 nolearning
ip link set vxlan0 master br-vxlan
ip link set vxlan0 up
```

### 控制面配置

控制面配置，这里的控制面是两个gobgpd进程（分别在as 65001和as 65002上）和一个自研的vxlan.py进程与这两个gobgpd交互，vxlan.py见这里 [链接](https://github.com/solofox/evpn-ha/blob/main/gobgp/vxlan.py)：

#### gobgp配置
两个gobgpd进程（分别在as 65001和as 65002上），它的配置简单，就是在本地监听grpc api与本地的vxlan.py交互，通过bgp协议和相应的RR交互：

```
[global.config]
  as = 65001
  router-id = "192.168.1.65" # 运行 GoBGP 的设备的 IP 地址
  port = -1

# 配置与 RR 的邻居关系
[[neighbors]]
  [neighbors.config]
    neighbor-address = "192.168.1.2" # RR 的 IP 地址
    peer-as = 65001                    # 与 RR 是 iBGP，AS 号相同
  [neighbors.transport.config]
    local-address = "192.168.1.65"     # 本机用于建立 BGP 会话的源 IP
  [neighbors.timers.config]
    hold-time = 9                    # 默认 Hold Time (秒)
    keepalive-interval = 3           # 默认 Keepalive Interval (秒)
    connect-retry = 3                # 默认连接重试间隔 (秒)
  # 启用 L2VPN EVPN 地址族
  [[neighbors.afi-safis]]
    [neighbors.afi-safis.config]
      afi-safi-name = "l2vpn-evpn"
    # (可选) 启用优雅重启
    [neighbors.afi-safis.mp-graceful-restart.config]
      enabled = true
```

#### vxlan.py

vxlan.py见这里 [链接](https://github.com/solofox/evpn-ha/blob/main/gobgp/vxlan.py)

## 一键部署 
[这里](https://github.com/solofox/evpn-ha/tree/main/gobgp)是一键部署terraform脚本，天下武功，为快不破！

## 验证

### 运行vxlan.py

启动日志：

```text
2025-09-04 06:52:47,863 [MainThread] INFO Found primary IPv4 address on eth0: 192.168.1.65/24
2025-09-04 06:52:47,863 [MainThread] INFO Interface 'eth0' - MAC: 00:16:3e:4d:2d:e8, Primary IP: 192.168.1.65
2025-09-04 06:52:47,865 [MainThread] INFO Found primary IPv4 address on eth1: 192.168.2.65/24
2025-09-04 06:52:47,865 [MainThread] INFO Interface 'eth1' - MAC: 00:16:3e:4d:2c:39, Primary IP: 192.168.2.65
2025-09-04 06:52:47,868 [MainThread] INFO Found primary IPv4 address on br-vxlan: 192.168.0.65/24
2025-09-04 06:52:47,868 [MainThread] INFO Interface 'br-vxlan' - MAC: 76:3f:65:0a:03:3b, Primary IP: 192.168.0.65
2025-09-04 06:52:47,868 [MainThread] INFO Configuration:
BRIDGE_IFNAME   = "br-vxlan"       192.168.0.65 76:3f:65:0a:03:3b
VXLAN_PORT_NAME = "vxlan0"
ROUTER_DEV1     = "eth0"
ROUTER_DEV2     = "eth1"
ASN1            = 65001
ROUTER_ID1      = "192.168.1.65"
ASN2            = 65002
ROUTER_ID2      = "192.168.2.65
VNI             = 100

2025-09-04 06:52:47,868 [MainThread] INFO Executing command: gobgp -p 50051 global -a evpn rib add macadv 76:3f:65:0a:03:3b 192.168.0.65 rd 192.168.1.65:100 rt 65001:100 etag 0 label 100 nexthop 192.168.1.65
2025-09-04 06:52:47,879 [MainThread] INFO EVPN Type-2 route announced successfully.
2025-09-04 06:52:47,879 [MainThread] INFO To Withdraw route: gobgp -p 50051 global -a evpn rib del macadv 76:3f:65:0a:03:3b 192.168.0.65 rd 192.168.1.65:100 etag 0 label 100 nexthop 192.168.1.65
2025-09-04 06:52:47,879 [MainThread] INFO Executing command: gobgp -p 50052 global -a evpn rib add macadv 76:3f:65:0a:03:3b 192.168.0.65 rd 192.168.2.65:100 rt 65002:100 etag 0 label 100 nexthop 192.168.2.65
2025-09-04 06:52:47,887 [MainThread] INFO EVPN Type-2 route announced successfully.
2025-09-04 06:52:47,887 [MainThread] INFO To Withdraw route: gobgp -p 50052 global -a evpn rib del macadv 76:3f:65:0a:03:3b 192.168.0.65 rd 192.168.2.65:100 etag 0 label 100 nexthop 192.168.2.65
2025-09-04 06:52:47,887 [MainThread] INFO EVPN listeners started.
2025-09-04 06:52:47,890 [grpc1-thread] INFO Bridge 'br-vxlan' index: 4
2025-09-04 06:52:47,890 [grpc2-thread] INFO Bridge 'br-vxlan' index: 4
2025-09-04 06:52:47,890 [grpc1-thread] INFO [grpc1] Starting EVPN Type-2 listener for 127.0.0.1:50051
2025-09-04 06:52:47,890 [grpc2-thread] INFO [grpc2] Starting EVPN Type-2 listener for 127.0.0.1:50052
2025-09-04 06:52:47,896 [grpc1-thread] INFO handle_route: nlri_data={'mac': '76:3f:65:0a:03:3b', 'ip': '192.168.0.65', 'vni': 100, 'rd': '192.168.1.65:100'}, nexthop=192.168.1.65
2025-09-04 06:52:47,896 [grpc1-thread] INFO ignore self route: {"ip": "192.168.0.65", "mac": "76:3f:65:0a:03:3b", "nexthop": "192.168.1.65", "rd": "192.168.1.65:100", "vni": 100, "creator": "thread1"}
2025-09-04 06:52:47,899 [grpc2-thread] INFO handle_route: nlri_data={'mac': '76:3f:65:0a:03:3b', 'ip': '192.168.0.65', 'vni': 100, 'rd': '192.168.2.65:100'}, nexthop=192.168.2.65
2025-09-04 06:52:47,899 [grpc2-thread] INFO ignore self route: {"ip": "192.168.0.65", "mac": "76:3f:65:0a:03:3b", "nexthop": "192.168.2.65", "rd": "192.168.2.65:100", "vni": 100, "creator": "thread2"}
2025-09-04 06:52:50,266 [grpc2-thread] INFO handle_route: nlri_data={'mac': '3a:36:8c:f7:2a:9c', 'ip': '192.168.0.87', 'vni': 100, 'rd': '192.168.2.87:100'}, nexthop=192.168.2.87
2025-09-04 06:52:50,266 [grpc2-thread] INFO announce route: current_route={"ip": "192.168.0.87", "mac": "3a:36:8c:f7:2a:9c", "nexthop": "192.168.2.87", "rd": "192.168.2.87:100", "vni": 100, "creator": "thread2"}, available_routes=[{"ip": "192.168.0.87", "mac": "3a:36:8c:f7:2a:9c", "nexthop": "192.168.2.87", "rd": "192.168.2.87:100", "vni": 100, "creator": "thread2"}], is_best_route=True
2025-09-04 06:52:50,267 [grpc2-thread] INFO Added neighbor (add): 192.168.0.87 -> 3a:36:8c:f7:2a:9c
2025-09-04 06:52:50,269 [grpc2-thread] INFO VTEP interface 'vxlan0' index: 5
2025-09-04 06:52:50,269 [grpc2-thread] INFO FDB added: MAC=3a:36:8c:f7:2a:9c, VNI=100, VTEP=192.168.2.87, on VXLAN-dev(vxlan0)
2025-09-04 06:52:54,888 [grpc1-thread] INFO handle_route: nlri_data={'mac': '3a:36:8c:f7:2a:9c', 'ip': '192.168.0.87', 'vni': 100, 'rd': '192.168.1.87:100'}, nexthop=192.168.1.87
2025-09-04 06:52:54,888 [grpc1-thread] INFO announce route: current_route={"ip": "192.168.0.87", "mac": "3a:36:8c:f7:2a:9c", "nexthop": "192.168.1.87", "rd": "192.168.1.87:100", "vni": 100, "creator": "thread1"}, available_routes=[{"ip": "192.168.0.87", "mac": "3a:36:8c:f7:2a:9c", "nexthop": "192.168.1.87", "rd": "192.168.1.87:100", "vni": 100, "creator": "thread1"}, {"ip": "192.168.0.87", "mac": "3a:36:8c:f7:2a:9c", "nexthop": "192.168.2.87", "rd": "192.168.2.87:100", "vni": 100, "creator": "thread2"}], is_best_route=True
2025-09-04 06:52:54,889 [grpc1-thread] INFO Neighbor 192.168.0.87 -> 3a:36:8c:f7:2a:9c already exists, try to replace if is_best_route True.
2025-09-04 06:52:54,890 [grpc1-thread] INFO Refreshed neighbor (replace): 192.168.0.87 -> 3a:36:8c:f7:2a:9c
2025-09-04 06:52:54,890 [grpc1-thread] INFO FDB entry for MAC=3a:36:8c:f7:2a:9c already exists.
2025-09-04 06:52:54,890 [grpc1-thread] INFO Best route changed or updated, replacing FDB for 3a:36:8c:f7:2a:9c.
2025-09-04 06:52:54,891 [grpc1-thread] INFO FDB replaced/refreshed: MAC=3a:36:8c:f7:2a:9c, VNI=100, VTEP=192.168.1.87
```

### 查看gobgp的路由状态

```bash
root@server-l:~# gobgp -p 50051 global rib  -a evpn
   Network                                                                            Labels     Next Hop             AS_PATH              Age        Attrs
*> [type:macadv][rd:192.168.1.65:100][etag:0][mac:76:3f:65:0a:03:3b][ip:192.168.0.65] [100]      192.168.1.65                              00:02:53   [{Origin: ?} {Extcomms: [65001:100]} [ESI: single-homed]]
*> [type:macadv][rd:192.168.1.87:100][etag:0][mac:3a:36:8c:f7:2a:9c][ip:192.168.0.87] [100]      192.168.1.87                              00:02:46   [{Origin: ?} {Med: 0} {LocalPref: 100} {Originator: 192.168.1.87} {ClusterList: [192.168.1.2]} {Extcomms: [65001:100]} [ESI: single-homed]]

root@server-l:~# gobgp -p 50052 global rib  -a evpn
   Network                                                                            Labels     Next Hop             AS_PATH              Age        Attrs
*> [type:macadv][rd:192.168.2.65:100][etag:0][mac:76:3f:65:0a:03:3b][ip:192.168.0.65] [100]      192.168.2.65                              00:03:07   [{Origin: ?} {Extcomms: [65002:100]} [ESI: single-homed]]
*> [type:macadv][rd:192.168.2.87:100][etag:0][mac:3a:36:8c:f7:2a:9c][ip:192.168.0.87] [100]      192.168.2.87                              00:03:04   [{Origin: ?} {Med: 0} {LocalPref: 100} {Originator: 192.168.2.87} {ClusterList: [192.168.2.2]} {Extcomms: [65002:100]} [ESI: single-homed]]
```

这个时候server-l同时向两个as都宣告了路由，也同时从两个as都收到了server-r（192.168.0.87）的type2路由。

### ping

```bash
# ping 192.168.0.87
PING 192.168.0.87 (192.168.0.87) 56(84) bytes of data.
64 bytes from 192.168.0.87: icmp_seq=1 ttl=64 time=0.465 ms
64 bytes from 192.168.0.87: icmp_seq=2 ttl=64 time=0.243 ms
^C
--- 192.168.0.87 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 0.243/0.354/0.465/0.111 ms
```

### 抓包

这时看到去玩192.168.0.87的路由走的是192.168.1.87这个VTEP。

```bash
# tcpdump -lenvvv -i any udp port 4789
tcpdump: data link type LINUX_SLL2
tcpdump: listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
06:59:15.422544 eth0  Out ifindex 2 00:16:3e:4d:2d:e8 ethertype IPv4 (0x0800), length 154: (tos 0x0, ttl 64, id 60678, offset 0, flags [none], proto UDP (17), length 134)
    192.168.1.65.48339 > 192.168.1.87.4789: [bad udp cksum 0x846c -> 0x6749!] VXLAN, flags [I] (0x08), vni 100
76:3f:65:0a:03:3b > 3a:36:8c:f7:2a:9c, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 1498, offset 0, flags [DF], proto ICMP (1), length 84)
    192.168.0.65 > 192.168.0.87: ICMP echo request, id 3155, seq 1, length 64
06:59:15.422929 eth0  In  ifindex 2 ee:ff:ff:ff:ff:ff ethertype IPv4 (0x0800), length 154: (tos 0x0, ttl 64, id 30793, offset 0, flags [none], proto UDP (17), length 134)
    192.168.1.87.34474 > 192.168.1.65.4789: [udp sum ok] VXLAN, flags [I] (0x08), vni 100
3a:36:8c:f7:2a:9c > 76:3f:65:0a:03:3b, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 45156, offset 0, flags [none], proto ICMP (1), length 84)
    192.168.0.87 > 192.168.0.65: ICMP echo reply, id 3155, seq 1, length 64
```

通过fdb也可以确认：
```bash
root@server-l:~# bridge fdb
01:00:5e:00:00:01 dev eth0 self permanent
33:33:00:00:00:01 dev eth0 self permanent
33:33:ff:4d:2d:e8 dev eth0 self permanent
01:80:c2:00:00:00 dev eth0 self permanent
01:80:c2:00:00:03 dev eth0 self permanent
01:80:c2:00:00:0e dev eth0 self permanent
33:33:00:00:00:01 dev eth1 self permanent
01:00:5e:00:00:01 dev eth1 self permanent
33:33:ff:4d:2c:39 dev eth1 self permanent
01:80:c2:00:00:00 dev eth1 self permanent
01:80:c2:00:00:03 dev eth1 self permanent
01:80:c2:00:00:0e dev eth1 self permanent
33:33:00:00:00:01 dev br-vxlan self permanent
01:00:5e:00:00:6a dev br-vxlan self permanent
33:33:00:00:00:6a dev br-vxlan self permanent
01:00:5e:00:00:01 dev br-vxlan self permanent
33:33:ff:0a:03:3b dev br-vxlan self permanent
76:3f:65:0a:03:3b dev br-vxlan vlan 1 master br-vxlan permanent
76:3f:65:0a:03:3b dev br-vxlan master br-vxlan permanent
9e:87:0e:a6:a9:3c dev vxlan0 vlan 1 master br-vxlan permanent
9e:87:0e:a6:a9:3c dev vxlan0 master br-vxlan permanent
3a:36:8c:f7:2a:9c dev vxlan0 dst 192.168.1.87 self extern_learn permanent

root@server-l:~# ip neigh get 192.168.0.87 dev br-vxlan
192.168.0.87 dev br-vxlan lladdr 3a:36:8c:f7:2a:9c extern_learn NOARP 
```

# 感想

当时坑还是蛮多的，网上有的资料也不甚详细。简单地说，就是需要让网络模型跟FRR定义的一致：*必须创建bridge，把VTEP加入到bridge中，业务接口使用bridge的三层口*。我最开始尝试的方法，都是像过去配置路由器那样，把业务IP配置在lo0上，然后想办法让bgpd把lo0的IP和MAC宣告出去，都失败了。

作为一个多年的基础架构研发工程师，多年的SDN从业者，还是更喜欢用软件来控制网络, 虽然写vxlan.py在AI copilot帮助下也花了些时间，SDN的口号是：**more software, more flexiable!**
