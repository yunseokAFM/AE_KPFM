import zmq
from src.core.SmartRemote import SmartRemote

class CHECKER(SmartRemote):

    def __init__(self):
        super().__init__()
        timeout = 1000 # ms
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt( zmq.LINGER, 0)
        self.socket.setsockopt( zmq.AFFINITY, 1)

    def heartbeat_connection(self):
        """Check connection"""
        self.set_method("js_connect")
        self.set_params("None")
        payload = self.generate_payload() 
        self.socket.send_json(payload)  
        try:
            reply = self.socket.recv_json() 
            value = 2
        except zmq.error.Again as e:
            value = 0
        finally: 
            self.socket.close()
            self.context.term()
            self.connect_socket()
        return value

    def abort_approach(self):
        """Abort approach"""
        self.set_method("js_cmd")
        self.set_params('spm.approach.stop();\n')
        payload = self.generate_payload() 
        self.socket.send_json(payload)
        print(payload)
        try:
            reply = self.socket.recv_json()
            value = True
            print(reply)
        except zmq.error.Again as e:
            value = False
            print(e)
        finally: 
            self.socket.close()
            self.context.term()
            self.connect_socket()
        return value

    def abort_scan(self):
        """Abort scan"""
        self.set_method("js_cmd")
        self.set_params('spm.scan.stop(;\n') 
        payload = self.generate_payload() 
        self.socket.send_json(payload)  
        try:
            reply = self.socket.recv_json() 
            value = True
        except zmq.error.Again as e:
            value = False
        finally: 
            self.socket.close()
            self.context.term()
            self.connect_socket()
        return value
