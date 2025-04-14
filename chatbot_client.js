    const startChat=()=>{ 

        socket.onopen = () => {
          console.log('Sunucuya bağlandı.');
        };
    
        socket.onmessage = (event) => {
          const list = $("#chatbotHistory");
          const messageContent= $(list).val();
          const message= event.data;
          $(list).val(messageContent + "\r\n" + message);
          $("#divChatbotHistory").append(setMessageDiv(message, false))
          $("#divChatbotHistory").animate({ scrollTop: 520 }, "slow");
        };
    }

    function sendMessage() {
      const input = $("#chatbotMessage");
      var message=$(input).val();
      socket.send(message);
      $("#divChatbotHistory").append(setMessageDiv(message, true))
      $(input).val("");
    }

    const setMessageDiv=(message, IsClient)=>{
      var cssClass= IsClient?"right":"left";
      let element=`<div style='width:300px; height:100px;' class='card ${cssClass}'>` + 
                    `<div class='card-body'>${message}</div>` +
                    `</div>`;
                    return element;
    }
    const socket = new WebSocket('ws://127.0.0.1:12345');
    startChat();