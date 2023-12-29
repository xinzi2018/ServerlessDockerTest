'''
	filename: zipper.py
	coding: utf-8
	author: Hao
	date: 2022/3/6
	brief: You can use this py file to zip a string data of picture tracks, 
			or import this py file in anothor py file to do the same work.
			The usage is in the __main__ part.
'''


'''
	obj: static class, 
		the method in this class can be used to zip the string data in anothor file or in main.
'''
class Zipper():
    '''
        func: data check,
                check whether the string type data of a picture has a right format.
    '''
    def raiseStrFormatError(string):  #4
        if string[0] != 'B' or string[-1] != 'D':
            raise Exception("String type data must begin with 'B' and end with 'D', But receive an wrong format.")
        if (len(string) - 2 + 1) % 14 != 0:
            raise Exception("The length of string type data is not divisible, please check your string data.")
            
    '''
        func: get the string type data of a picture,
                and separate each part of a single track.
    '''
    def getTracksStr(data_path):  #3
        with open(data_path, 'r') as f:
            string = f.read()
            f.close()

        Zipper.raiseStrFormatError(string)  # data check

        string = string[1:-1]  # Remove 'B' and 'D'

        part_list = list()  # Parts of track (string type)
        for i in range(0, len(string), 14):
            part_list.append(string[i:i+13])

        return part_list


    '''
        func: generate Track instances list with class 'Track'.
    '''
    def generateTrackInstancesList(data_path):  #2
        track_list = list()  # Parts of track (class type)

        for enum in Zipper.getTracksStr(data_path):
            track_list.append(Track(int(enum[-1]), enum[0:6], enum[6:12]))

        return track_list

    '''
        func: generate string data after zip.
    '''
    def zipStrData(data_path):  #1
        return_str = str()

        return_str += '{{{{{'
        for track in Zipper.generateTrackInstancesList(data_path):
            print(track.con_str+"||||"+track.len_str)
            return_str += track.con_str
            return_str += track.len_str
        return_str += '}}}}}'
        
        return return_str

    '''
        func: save string to a file.
    '''
    def saveTo(data, save_path):
        with open(save_path, 'w') as f:
            f.write(data)
            f.close()
    '''
    obj: stand for a single track of a picture
    '''
class Track():
    def __init__(self, mode_str, x_str, y_str):   ##6
        '''
            aim: process the str type value to int type value
        '''
        self.mode = int(mode_str)  # Pen updown mode
        self.x = int(x_str)  # X length
        self.y = int(y_str)  # Y length

        '''
            aim: separate the direction and the length
        '''
        self.x_len = abs(self.x)
        self.y_len = abs(self.y)
        self.x_dir = 1 if self.x >= 0 else -1
        self.y_dir = 1 if self.y >= 0 else -1

        '''
            aim: value check before generate the zipped chars
        '''
        self.raiseValueError()  # Check if these values have wrong format

        '''
            aim: generate the chars after zip
        '''
        self.con_str = self.zipSelfCon()  # control char after zip
        self.len_str = self.zipSelfLen()  # length chars after zip

    '''
        func: raise error when the input value is unallowed
    '''
    def raiseValueError(self):  #7
        if self.mode < 0 or self.mode > 3:
            raise Exception("Pen updown mode should belong in 0 to 3, But receive a {}.".format(self.mode))
        if self.x_len > (94*94) or self.y_len > (94*94):
            raise Exception("The max length of a single move is 8836, But receive a ({}, {}).".format(self.x_len, self.y_len))

    '''
        func: zip the 'control' char of a pack
        brief: the first three bits of the 'control' char is '010'
    '''
    def zipSelfCon(self):   #8
        con = 0b01000000

        if self.x_dir > 0:
            con |= 0b00010000
        if self.y_dir > 0:
            con |= 0b00001000
        con |= self.mode

        return chr(con)

    '''
        func: zip the 'length' chars of a pack
        brief: there are two chars in a single direction,
                the ASCII code of the char scattered between 20H(Space) and 7EH(Tilde),
                so the max length of a single move is 94x94=8836
    '''
    def zipSelfLen(self):  #9
        len_chars = str()

        len_chars += chr(int(self.x_len / 94 + 32))
        len_chars += chr(int(self.x_len % 94 + 32))
        len_chars += chr(int(self.y_len / 94 + 32))
        len_chars += chr(int(self.y_len % 94 + 32))

        return len_chars



if __name__ == '__main__':

    zipped_string = Zipper.zipStrData('data.txt')  # Zip the original string data of picture track
    Zipper.saveTo(zipped_string, 'data_zipped.txt')  # Save the zipped data to another txt file

    # Debug
#    print(zipped_string)
    print("Length is: {}".format(len(zipped_string)))
    # Debug End
