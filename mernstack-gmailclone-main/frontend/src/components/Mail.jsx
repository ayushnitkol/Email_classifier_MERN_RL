import React from 'react';
import { IoMdArrowBack, IoMdMore } from 'react-icons/io';
import { useNavigate } from 'react-router-dom';
import { BiArchiveIn } from "react-icons/bi";
import { MdDeleteOutline, MdKeyboardArrowLeft, MdKeyboardArrowRight, MdOutlineAddTask, MdOutlineDriveFileMove, MdOutlineMarkEmailUnread, MdOutlineReport, MdOutlineWatchLater } from 'react-icons/md';
import { useSelector } from 'react-redux';
import axios from 'axios';
import toast from 'react-hot-toast';

const Mail = () => {
  const navigate = useNavigate();
  const { selectedEmail } = useSelector(store => store.app);

  const deleteHandler = async () => {
    try {
      const res = await axios.delete(`http://localhost:8080/api/v1/email/${selectedEmail?._id}`, { withCredentials: true });
      toast.success(res.data.message);
      navigate("/");
    } catch (error) {
      console.error(error);
      toast.error("Failed to delete");
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-gray-600">
          <div onClick={() => navigate("/")} className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><IoMdArrowBack /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><BiArchiveIn /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><MdOutlineReport /></div>
          <div onClick={deleteHandler} className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><MdDeleteOutline /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><MdOutlineMarkEmailUnread /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><MdOutlineWatchLater /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><MdOutlineAddTask /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><MdOutlineDriveFileMove /></div>
          <div className="p-2 rounded-full hover:bg-gray-100 cursor-pointer"><IoMdMore /></div>
        </div>

        <div className="flex items-center gap-3 text-sm text-gray-600">
          <span>1–50</span>
          <MdKeyboardArrowLeft size="20px" />
          <MdKeyboardArrowRight size="20px" />
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-800">{selectedEmail?.subject}</h1>
            <div className="text-sm text-gray-500 mt-2">{selectedEmail?.to} <span className="ml-2 text-gray-400">to me</span></div>
          </div>
          <div className="text-sm text-gray-400">{new Date(selectedEmail?.createdAt).toLocaleString()}</div>
        </div>

        <div className="mt-6 text-gray-700 whitespace-pre-wrap">
          {selectedEmail?.message}
        </div>
      </div>
    </div>
  );
};

export default Mail;
