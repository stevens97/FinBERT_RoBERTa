def fetch_Articles(write_path, df_news):
    for i in range(len(df_news)):
        temp = None
        print(df_news['URL'][i])
        print(i)

        URL = df_news['URL'][i].replace('/', '_').replace('\\', '_').replace(':', '_')

        file = r'{}.txt'.format(URL)
        file_path = r'{}/{}'.format(write_path, file)

        if os.path.exists(file_path) == False:
            temp = call(get_full_content, timeout=10, URL=df_news['URL'][i])

            print(temp)

    return None