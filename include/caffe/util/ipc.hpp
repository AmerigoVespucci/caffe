/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ipc.hpp
 * Author: eli
 *
 * Created on March 14, 2016, 9:25 PM
 */

#ifndef IPC_HPP
#define IPC_HPP

using boost::asio::ip::tcp;

tcp::socket* IPCServerInit(const char * port_str);
tcp::socket* IPCClientInit(const char * host, int port_num);
int CaffeIPCSendMsg(tcp::socket& socket, CaffeIpc& Msg) ;
int CaffeIPCRcvMsg(tcp::socket& socket, CaffeIpc& Msg);
void IPCClientClose(tcp::socket* socket);

#endif /* IPC_HPP */

