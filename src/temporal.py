"""

"""
s_m = 60
s_h = s_m * 60
s_d = s_h * 24
speryr = s_d * 365

mo_30s = 30 * s_d
mo_31s = 31 * s_d

month_dict = {1: mo_31s, 
              2: 28 * s_d, 
              3: mo_31s, 
              4: mo_30s, 
              5: mo_31s, 
              6: mo_30s, 
              7: mo_30s, 
              8: mo_30s, 
              9: mo_30s, 
              10: mo_30s, 
              11: mo_31s, 
              12: mo_30s}

month_name = {1: 'January', 
              2: 'February', 
              3: 'March', 
              4: 'April', 
              5: 'May', 
              6: 'June', 
              7: 'July', 
              8: 'August', 
              9: 'September', 
              10: 'October', 
              11: 'November', 
              12: 'December'}


def float_sec_to_time_dict(tiq):
    time_dict = {}
    date_str = ''
    s = int(tiq)
    ms = tiq - s
    
    Y = s // speryr
    s = s - Y * speryr
    leap_years = Y // 4
    s = s - leap_years * s_d
    Y = 1970 + Y
    time_dict['year'] = Y
    
    for month_number, month_seconds in month_dict.items():
        if s > month_seconds:
            s = s - month_seconds
        else:
            print(month_number)
            break
            
    Mnth = month_number
    time_dict['month'] = Mnth
    
    D = s // s_d
    s = s - s_d * D
    D += 1
    time_dict['day'] = D
    
    tz = time.timezone
    tz_correction = tz // 60**2
    H = s // s_h
    s = s - H * s_h
    H = H - tz_correction + 1
    time_dict['hour'] = H
    
    mnts = s // s_m
    s = s - mnts * s_m
    
    time_dict['minutes'] = mnts
    time_dict['seconds'] = s + ms
    
    time_dict['string'] = '%i/%i/%i_%i:%i:%0.2f'%(Mnth, D, Y, H, mnts, s + ms)
    time_dict['stamp'] = '%i_%i_%i_%i_%i_%0.12f'%(Y, Mnth, D, H, mnts, s + ms)
    time_dict['printy_string'] = '%i:%i:%02i %s %i, %i'%(H, mnts, s, month_name[Mnth], D, Y)
    
    return time_dict

# time_dict = float_sec_to_time_dict(time.time())
# time_dict['stamp'], time_dict['printy_string'], time_dict['string']
