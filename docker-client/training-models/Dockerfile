from mlm-pipe-ubuntu24.04-cuda12.04:mlm

RUN apt-get update  -y &&\
    apt-get install -y rclone\
		       jq\
		       jo

RUN apt-get update  -y &&\
    apt-get install -y --reinstall ca-certificates
 
COPY ./rclone.conf /root/.config/rclone/rclone.conf

COPY .env /
COPY create_dict_var.sh /
COPY client.sh /
RUN chmod +x client.sh
COPY debugger.sh /
RUN chmod +x debugger.sh

ENTRYPOINT ["/client.sh"]
# CMD ["train_bert_on_cleaned_efcamdat_all.json"]
