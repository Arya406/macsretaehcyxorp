import time
print("Starting test script...")
with open('test_output.txt', 'w') as f:
    f.write('Test successful at ' + time.ctime() + '\n')
print("Test file created")
time.sleep(5)  # Keep the script running for 5 seconds
