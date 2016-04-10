#include <cstdlib>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <glog/logging.h>

#include "caffe/proto/ipc.pb.h"
#include "caffe/util/ipc.hpp"

using namespace std;
using std::string;

using boost::asio::ip::tcp;

tcp::socket* IPCServerInit(const char * port_str) {
	boost::asio::io_service io_service;

	int port_num = atoi(port_str);
	tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port_num));

	//boost::shared_ptr<tcp::socket> socket(new tcp::socket(io_service));
	tcp::socket* socket = new tcp::socket(io_service);
	acceptor.accept(*socket);
	
	LOG(INFO) << "Server connect request received\n";

	return socket;
}

//boost::shared_ptr<tcp::socket> ClientInit(const char * host, int port_num) {
tcp::socket* IPCClientInit(const char * host, int port_num) {
	boost::asio::io_service io_service;

	tcp::resolver resolver(io_service);
	//string port_str = to_string(1543);
	stringstream ssport;
	ssport << port_num;
	tcp::resolver::query query(host, ssport.str());
	tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
	tcp::resolver::iterator end;

	//tcp::socket socket(io_service);
	//boost::shared_ptr<tcp::socket> socket(new tcp::socket(io_service));
	tcp::socket* socket = new tcp::socket(io_service);
	boost::system::error_code error = boost::asio::error::host_not_found;
	while (error && endpoint_iterator != end) {
		socket->close();
		socket->connect(*endpoint_iterator++, error);
	}
	if (error) {
		//throw boost::system::system_error(error);
		cerr << "CaffeIPC socket error: " << error << endl;
		return NULL;
	}

	
	LOG(INFO) << "Client connection established\n";

	return socket;
}

void IPCClientClose(tcp::socket* socket) {
	socket->close();
}

int CaffeIPCSendMsg(tcp::socket& socket, CaffeIpc& Msg) {
	int SerializedSize = Msg.ByteSize();
	unsigned char * data_buf = new unsigned char[SerializedSize];
	Msg.SerializeToArray(data_buf, SerializedSize);
	boost::array<int, 1> size_buf;
	size_buf[0] = SerializedSize;
	boost::system::error_code error;
	int len = write(	socket, boost::asio::buffer(size_buf),
						boost::asio::transfer_exactly(sizeof(int)), error);
	len = write(socket, boost::asio::buffer(data_buf, SerializedSize), 
				boost::asio::transfer_exactly(size_buf[0]), error);
	delete[] data_buf;
	if (error != 0) {
		std::cerr << "Error sending IPC message.\n";
	}
//	if (error == boost::asio::error::eof)
//		return -1; // Connection closed cleanly by peer.
//	else if (error)
//		throw boost::system::system_error(error); // Some other error.
	return len;
}

int CaffeIPCRcvMsg(tcp::socket& socket, CaffeIpc& Msg) {
	boost::system::error_code error;
	boost::array<int, 1> size_buf;
	int len = read(socket, boost::asio::buffer(size_buf), boost::asio::transfer_exactly(sizeof(int)), error);
	if (error == 0) {
		int alloc_size = size_buf[0];
		LOG(INFO) << "Receiving a msg of size " << alloc_size << std::endl;
		char * data_buf = new char[alloc_size];
		len = read(	socket, boost::asio::buffer(data_buf, alloc_size), 
					boost::asio::transfer_exactly(alloc_size ), error);
		LOG(INFO) << "Read " << len << " bytes\n";
		if (error == 0) {
			Msg.ParseFromArray(data_buf, alloc_size);
		}
		delete[] data_buf;
			
	}

	if (error != 0) {
		cerr << "Error receiving IPC message.\n";
		return -1;
	}
//	if (error == boost::asio::error::eof)
//		return -1; // Connection closed cleanly by peer.
//	else if (error)
//		throw boost::system::system_error(error); // Some other error.
	
	return len;
}








