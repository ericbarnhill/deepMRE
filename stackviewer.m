setappdata(hFig,'MyMatrix',stack); %// You could use %//setappdata(0,'MyMatrix',MyMatrix) to store in the base workspace. 

%// Display 1st frame
imshow(stack(:,:,:,1))

%// IMPORTANT. Update handles structure.
guidata(hFig,handles);

%// Listener callback, executed when you drag the slider.

    function XListenerCallBack

        %// Retrieve handles structure. Used to let MATLAB recognize the
        %// edit box, slider and all UI components.
        handles = guidata(gcf);

%// Here retrieve MyMatrix using getappdata.
MyMatrix = getappdata(hFig,'MyMatrix');

        %// Get current frame
        CurrentFrame = round((get(handles.SliderFrame,'Value')));
        set(handles.Edit1,'String',num2str(CurrentFrame));

        %// Display appropriate frame.
        imshow(MyMatrix(:,:,:,CurrentFrame),'Parent',handles.axes1);

        guidata(hFig,handles);
    end


%// Slider callback; executed when the slider is release or you press
%// the arrows.
    function XSliderCallback(~,~)

        handles = guidata(gcf);

%// Here retrieve MyMatrix using getappdata.
    MyMatrix = getappdata(hFig,'MyMatrix');

        CurrentFrame = round((get(handles.SliderFrame,'Value')));
        set(handles.Edit1,'String',num2str(CurrentFrame));

        imshow(MyMatrix(:,:,:,CurrentFrame),'Parent',handles.axes1);

        guidata(hFig,handles);
    end

end