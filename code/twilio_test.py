from twilio.rest import Client

# Find these values at https://twilio.com/user/account
account_sid = "ACccc6b5c88322691206e98a423af622e2"
auth_token = "6aa66d8f3c4c3894fd7de9a1422e08cf"

client = Client(account_sid, auth_token)

client.api.account.messages.create(
    to="+91-8073005889",
    from_="+12053510697",
    body="Hello there!")
