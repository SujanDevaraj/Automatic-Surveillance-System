#ACca7baac389b4fbbb972b2501cca55c6a
#515fb46cbb5297413b1ecc6010b22e2c
#+17622487761

from twilio.rest import Client 
 
account_sid = 'ACca7baac389b4fbbb972b2501cca55c6a' 
auth_token = '515fb46cbb5297413b1ecc6010b22e2c' 
client = Client(account_sid, auth_token) 
 
message = client.messages.create(  
                              from_='+17622487761', 
                              body='jj',      
                              to='+918884535386' 
                          ) 
 
print(message.sid)
