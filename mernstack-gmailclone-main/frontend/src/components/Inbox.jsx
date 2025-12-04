import React, { useState } from 'react';
import { MdCropSquare, MdInbox, MdKeyboardArrowLeft, MdKeyboardArrowRight } from 'react-icons/md';
import { FaCaretDown, FaUserFriends } from "react-icons/fa";
import { IoMdMore, IoMdRefresh } from 'react-icons/io';
import { GoTag } from "react-icons/go";
import Emails from './Emails';

const mailType = [
  { icon: <MdInbox size={'18px'} />, text: "Primary" },
  { icon: <GoTag size={'18px'} />, text: "Promotions" },
  { icon: <FaUserFriends size={'18px'} />, text: "Social" },
];

const Inbox = () => {
  const [selected, setSelected] = useState(0);

  return (
    <div className="flex-1">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 p-2 rounded-md hover:bg-gray-100 cursor-pointer">
              <MdCropSquare size={'18px'} />
              <FaCaretDown size={'14px'} />
            </div>
            <div className="p-2 rounded-md hover:bg-gray-100 cursor-pointer">
              <IoMdRefresh size={'18px'} />
            </div>
            <div className="p-2 rounded-md hover:bg-gray-100 cursor-pointer">
              <IoMdMore size={'18px'} />
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm text-gray-600">
            <span>1–50</span>
            <MdKeyboardArrowLeft size="20px" className="cursor-pointer hover:text-gray-800" />
            <MdKeyboardArrowRight size="20px" className="cursor-pointer hover:text-gray-800" />
          </div>
        </div>

        <div className="flex gap-3 mb-3">
          {mailType.map((item, index) => (
            <button
              key={index}
              onClick={() => setSelected(index)}
              className={`px-4 py-2 rounded-lg ${selected === index ? "border-b-4 border-b-indigo-600 text-indigo-600" : "text-gray-600 hover:bg-gray-50"}`}
            >
              <div className="flex items-center gap-2">{item.icon}<span>{item.text}</span></div>
            </button>
          ))}
        </div>

        <div className="max-h-[65vh] overflow-y-auto">
          <Emails />
        </div>
      </div>
    </div>
  );
};

export default Inbox;
