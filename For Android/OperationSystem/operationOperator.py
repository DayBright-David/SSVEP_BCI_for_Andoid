import datetime


class OperationState:
    def __init__(self):

        self.steaming_state = 'INIT'

        self.control_state = 'INIT'

        self.current_detect_state = 'INIT'


class operationOperator:

    def __init__(self):
        self.messenager = None
        self.streaming = None
        self.state = OperationState()
        self.events = ['CTNS', #connectToNeuroScan
                       'STAR', #startReceivedData
                       'STOP', #stopReceivedData
                       'STON', #StartOperation
                       'STRD', #StartRealTimeDetection
                       'EXIT'] #Exit Program

    def do_CTNS(self, event):
        self.streaming.connect()
        self.state.steaming_state = 'CTNS'
        message = 'CTOK'
        self.messenager.send_exchange_message(message)
        print('成功连接到NeuroScan服务器')

    def do_STAR(self, event):
        self.state.steaming_state = 'STAR'
        message = 'TROK'
        self.messenager.send_exchange_message(message)
        print('开始从NeuroScan服务器接收数据')


    def do_STOP(self, event):
        self.streaming.stop_receive_data()
        self.state.steaming_state = 'STOP'
        print('停止从NeuroScan服务器接收数据')
        #self.messenager.sendExchangeMessage(message)

    def do_STON(self, event):
        self.state.control_state = 'STON'
        print('设置为开始数据处理状态')
        message = 'TNOK'
        self.messenager.send_exchange_message(message)


    def do_EXIT(self, event):
        self.state.control_state = 'EXIT'
        print('处理程序准备退出')

    def do_STRD(self, event):
        self.state.current_detect_state = 'STRD'
        print('\n准备进入实时处理模式,%s'%datetime.datetime.now())

    def add_listener(self, event_manager):
        event_manager.AddEventListener('CTNS', self.do_CTNS)
        event_manager.AddEventListener('STAR', self.do_STAR)
        event_manager.AddEventListener('STOP', self.do_STOP)
        event_manager.AddEventListener('STON', self.do_STON)
        event_manager.AddEventListener('STRD', self.do_STRD)
        event_manager.AddEventListener('EXIT', self.do_EXIT)

    def remove_listener(self, event_manager):
        event_manager.RemoveEventListener('CTNS', self.do_CTNS)
        event_manager.RemoveEventListener('STAR', self.do_STAR)
        event_manager.RemoveEventListener('STOP', self.do_STOP)
        event_manager.RemoveEventListener('STON', self.do_STON)
        event_manager.RemoveEventListener('STRD', self.do_STRD)
        event_manager.RemoveEventListener('EXIT', self.do_EXIT)
